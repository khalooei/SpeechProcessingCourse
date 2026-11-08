"""
Mini-Task 9: Background Classification and Voice Activity Detection on VOiCES
=============================================================================

Course   : Digital Speech Processing - Dr. Khalooei
Student  : Ghassan Alsafar (404131085)

What this script does (end to end, one command):
  1. Downloads a small official subset of the VOiCES corpus
     (Lab41/VOiCES-subset on GitHub: foreground speaker at 90 degrees, all four
     background conditions) - only as many archives as needed.
  2. Reads the labels from the VOiCES file names
     (room, distractor = none / babb / musi / tele, speaker, mic, ...).
  3. Part A - Background (event) classification: which distractor is playing?
     Clip features (MFCC, spectral shape, and the same features measured only in
     the low-energy "pause" frames where the background is exposed), 6 classifiers,
     cross-validation grouped by speaker.
  4. Part B - Speech / non-speech and Voice Activity Detection:
     * reference labels come from the quiet ("none") recording of the same
       sentence, transferred to the noisy recordings by cross-correlation
       alignment (the same LibriSpeech sentence was replayed in every condition);
     * three VADs are compared per background condition:
       energy VAD, spectral (energy + flatness) VAD and a learned frame
       classifier (Random Forest on MFCC frames), plus Silero VAD if PyTorch
       is installed.
  5. Saves figures to ./figures, numbers to ./results and writes README.md.

Run:
    pip install -r requirements.txt
    python voices_background_vad.py
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import tarfile
import time
import urllib.request
from collections import Counter, defaultdict
from math import gcd
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import correlate, resample_poly
from scipy.signal import spectrogram as scipy_spectrogram

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sklearn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
RES_DIR = HERE / "results"
DATA_DIR = HERE / "data"

SUBSET_REPO = "https://github.com/Lab41/VOiCES-subset/raw/master/"
SUBSET_TARS = [f"VOiCES_90_{i}.tar" for i in range(1, 9)]

SR = 16000
CLASSES = ["none", "babb", "musi", "tele"]
CLASS_NAMES = {"none": "None (quiet)", "babb": "Babble", "musi": "Music", "tele": "TV"}
SEED = 42
FRAME, HOP = 0.025, 0.010                     # 25 ms frames, 10 ms hop

NAME_RE = re.compile(
    r"Lab41-SRI-VOiCES-(?P<room>rm\d+)-(?P<dist>none|babb|musi|tele)-sp(?P<spk>\d+)"
    r"-ch(?P<ch>\d+)-s(?:e)?g(?P<seg>\d+)-mc(?P<mic>\d+)-(?P<mtype>[a-z]+)-(?P<mloc>[a-z0-9]+)"
    r"(?:-dg(?P<deg>\d+))?", re.I)

# Chart style (validated categorical palette, fixed order)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
C_CLASS = {c: PALETTE[i] for i, c in enumerate(CLASSES)}
INK, INK_2, GRID = "#1f1f1e", "#5f5e58", "#e4e3dc"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2,
})


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def parse_name(stem: str):
    m = NAME_RE.search(stem)
    if not m:
        return None
    d = m.groupdict()
    d["dist"] = d["dist"].lower()
    d["sentence"] = f"sp{d['spk']}-ch{d['ch']}-sg{d['seg']}"
    d["key"] = f"{d['room']}-{d['sentence']}"          # same sentence, any microphone
    return d


def wav_files(root: Path):
    return [p for p in root.rglob("*.wav") if parse_name(p.stem)]


def class_counts(files):
    return Counter(parse_name(p.stem)["dist"] for p in files)


def _use_system_certificates():
    """On some Windows networks (VPN / antivirus / proxy) Python rejects GitHub's
    certificate although the browser accepts it. `truststore` (shipped with
    Anaconda) makes Python use the Windows certificate store, like the browser."""
    try:
        import truststore
        truststore.inject_into_ssl()
        return True
    except Exception:
        return False


def download(url: str, dest: Path):
    _use_system_certificates()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got, last = 0, 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if got - last > 20 * (1 << 20):
                last = got
                log(f"    {got / 2**20:6.0f} MB" + (f" / {total / 2**20:.0f} MB" if total else ""))
    tmp.replace(dest)


def safe_extract(tar_path: Path, out_dir: Path, depth: int = 0):
    """Extract only .wav files (and archives nested inside the archive)."""
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if m.isfile()
                   and m.name.lower().endswith((".wav", ".tar", ".tar.gz", ".tgz"))
                   and not m.name.startswith(("/", "..")) and ".." not in Path(m.name).parts]
        try:
            tar.extractall(out_dir, members=members, filter="data")
        except TypeError:                                   # Python < 3.12
            tar.extractall(out_dir, members=members)
    if depth < 2:
        for m in members:
            inner = out_dir / m.name
            if inner.suffix != ".wav" and inner.exists() and tarfile.is_tarfile(inner):
                safe_extract(inner, inner.parent, depth + 1)
                inner.unlink()


def ensure_dataset(data_dir: Path, min_per_class: int, max_tars: int) -> Path:
    """Use .wav files already in data_dir; otherwise download subset archives one by one
    until every background class has at least `min_per_class` recordings."""
    data_dir.mkdir(parents=True, exist_ok=True)
    ex_dir = data_dir / "extracted"
    # archives downloaded with the browser land in Downloads - pick them up automatically
    import shutil
    for cand in sorted((Path.home() / "Downloads").glob("VOiCES_90_*.tar")):
        target = data_dir / re.sub(r"\s*\(\d+\)", "", cand.name)
        if not target.exists() and not (ex_dir / f".{target.stem}.done").exists():
            log(f"  found {cand.name} in Downloads - moving it to data/")
            shutil.move(str(cand), str(target))
    for tar_path in sorted(data_dir.glob("*.tar")):          # archives copied in by hand
        flag = ex_dir / f".{tar_path.stem}.done"
        if not flag.exists() and tarfile.is_tarfile(tar_path):
            log(f"  extracting {tar_path.name} ...")
            safe_extract(tar_path, ex_dir)
            flag.parent.mkdir(parents=True, exist_ok=True)
            flag.touch()

    def enough():
        c = class_counts(wav_files(data_dir))
        return all(c.get(k, 0) >= min_per_class for k in CLASSES), c

    ok, c = enough()
    for name in SUBSET_TARS[:max_tars]:
        if ok:
            break
        dest = data_dir / name
        if (ex_dir / f".{Path(name).stem}.done").exists():
            continue
        log(f"  downloading {name} (official VOiCES subset, GitHub) ...")
        try:
            download(SUBSET_REPO + name, dest)
        except Exception as e:
            log(f"    failed: {e}")
            continue
        if not tarfile.is_tarfile(dest):
            log("    not a valid archive - skipped")
            continue
        log(f"  extracting {name} ...")
        safe_extract(dest, ex_dir)
        (ex_dir / f".{Path(name).stem}.done").touch()
        dest.unlink()                                   # keep the disk small
        ok, c = enough()
        log("  recordings so far: " + ", ".join(f"{k}={c.get(k, 0)}" for k in CLASSES))
    if sum(c.values()) == 0:
        sys.exit(
            "\nNo VOiCES recordings found and the automatic download did not work.\n"
            f"Download one or more archives by hand from\n  {SUBSET_REPO.replace('/raw/', '/blob/')}"
            "VOiCES_90_1.tar\nput the .tar file(s) inside\n  "
            f"{data_dir}\nand run this script again.")
    return data_dir


def load_wav(path: Path, max_sec: float):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    x = x.astype(np.float32)
    if sr != SR:
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return x[: int(max_sec * SR)]


def pick_files(files, per_class, rng):
    """Balanced selection, preferring sentences that exist in several conditions
    (needed for the VAD reference) and spreading over speakers."""
    info = [(p, parse_name(p.stem)) for p in files]
    by_key = defaultdict(set)
    for _, d in info:
        by_key[d["key"]].add(d["dist"])
    chosen = []
    for c in CLASSES:
        cand = [(p, d) for p, d in info if d["dist"] == c]
        rng.shuffle(cand)
        cand.sort(key=lambda t: -len(by_key[t[1]["key"]]))    # multi-condition first
        by_spk = defaultdict(list)
        for p, d in cand:
            by_spk[d["spk"]].append((p, d))
        sel, spks = [], list(by_spk)
        while len(sel) < per_class and any(by_spk.values()):    # round-robin over speakers
            for s in spks:
                if by_spk[s] and len(sel) < per_class:
                    sel.append(by_spk[s].pop(0))
        chosen += sel
    return chosen


# --------------------------------------------------------------------------
# 2. Features
# --------------------------------------------------------------------------
def frame_signal(x, fl, hp):
    if len(x) < fl:
        x = np.pad(x, (0, fl - len(x)))
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx]


def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)


_FB: dict = {}


def mel_fb(n_fft=512, n_mels=40):
    if (n_fft, n_mels) in _FB:
        return _FB[(n_fft, n_mels)]
    mels = np.linspace(0, hz_to_mel(SR / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / SR).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(r - c, 1)
    _FB[(n_fft, n_mels)] = fb
    return fb


def frame_features(x):
    """Per-frame descriptors (10 ms hop): log energy, 13 MFCC, spectral centroid,
    flatness, roll-off and zero-crossing rate. MFCC implemented from scratch."""
    fl, hp, n_fft = int(FRAME * SR), int(HOP * SR), 512
    raw = frame_signal(x, fl, hp)
    xe = np.append(x[0], x[1:] - 0.97 * x[:-1])
    fr = frame_signal(xe, fl, hp) * np.hamming(fl)
    P = (np.abs(np.fft.rfft(fr, n_fft)) ** 2) / n_fft + 1e-12
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    log_mel = np.log(P @ mel_fb(n_fft).T + 1e-10)
    mf = dct(log_mel, type=2, axis=1, norm="ortho")[:, :13]
    energy = 10 * np.log10(np.mean(raw ** 2, axis=1) + 1e-10)
    centroid = (P @ freqs) / P.sum(1)
    flat = np.exp(np.mean(np.log(P), 1)) / np.mean(P, 1)
    cum = np.cumsum(P, 1)
    rolloff = freqs[np.argmax(cum >= 0.85 * cum[:, -1:], axis=1)]
    zcr = np.mean(np.abs(np.diff(np.sign(raw), axis=1)) > 0, axis=1)
    Pr = (np.abs(np.fft.rfft(raw * np.hamming(fl), n_fft)) ** 2)        # no pre-emphasis
    band = (freqs >= 200) & (freqs <= 5000)
    return {"energy": energy, "mfcc": mf, "centroid": centroid / 1000, "flatness": flat,
            "rolloff": rolloff / 1000, "zcr": zcr, "P": Pr[:, band].astype(np.float32)}


def clip_vector(F):
    """Fixed-length clip descriptor for background classification.
    Statistics over all frames + statistics over the quietest 30 % of frames
    (pauses between words, where the background is heard on its own)."""
    e = F["energy"]
    M = np.column_stack([F["mfcc"], F["centroid"], F["flatness"], F["rolloff"], F["zcr"]])
    q = e <= np.percentile(e, 30)
    d = np.gradient(F["mfcc"], axis=0)
    return np.concatenate([
        M.mean(0), M.std(0), d.std(0),
        M[q].mean(0), M[q].std(0),
        [np.percentile(e, 10), np.percentile(e, 90), np.percentile(e, 90) - np.percentile(e, 10)],
    ])


def frame_matrix(F, ctx=5):
    """Per-frame features with +-ctx frame context averaging (for the learned VAD)."""
    M = np.column_stack([F["energy"] - np.median(F["energy"]), F["mfcc"],
                         F["centroid"], F["flatness"], F["zcr"]])
    k = np.ones(2 * ctx + 1) / (2 * ctx + 1)
    S = np.column_stack([np.convolve(M[:, j], k, mode="same") for j in range(M.shape[1])])
    D = np.column_stack([np.convolve(np.abs(np.gradient(M[:, j])), k, mode="same")
                         for j in range(1, 14)])
    return np.hstack([M, S, D])


# --------------------------------------------------------------------------
# 3. VAD helpers
# --------------------------------------------------------------------------
def smooth_mask(mask, min_speech=0.10, min_gap=0.25, hang=0.15):
    """Fill short gaps, drop short blobs, add hangover (all in seconds)."""
    m = mask.astype(bool).copy()
    fr = lambda s: max(1, int(round(s / HOP)))          # noqa: E731
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    for a, b in zip(en[:-1], st[1:]):                    # gaps
        if b - a < fr(min_gap):
            m[a:b] = True
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    for a, b in zip(st, en):                             # short blobs
        if b - a < fr(min_speech):
            m[a:b] = False
    if hang > 0:
        m = np.convolve(m.astype(float), np.ones(fr(hang)), mode="full")[: len(m)] > 0
    return m


def energy_vad(F, frac=0.45):
    e = np.convolve(F["energy"], np.ones(5) / 5, mode="same")
    lo, hi = np.percentile(e, 10), np.percentile(e, 99)
    return smooth_mask(e > lo + frac * (hi - lo))


def ltsd_vad(F, order=6, frac=0.35):
    """Long-Term Spectral Divergence VAD (Ramirez et al., 2004).
    The noise spectrum N(k) is estimated from the quietest 20 % of frames; for each
    frame the long-term spectral envelope LTSE(k) = max |X(k)| over +-`order`
    frames is compared with it:  LTSD = 10 log10( mean_k LTSE(k)^2 / N(k) ).
    Speech raises the divergence in many bands at once; stationary or slowly
    varying background does not."""
    P = F["P"].astype(np.float64)
    e = F["energy"]
    noise = P[e <= np.percentile(e, 20)].mean(0) + 1e-12
    from scipy.ndimage import maximum_filter1d
    ltse = maximum_filter1d(P, size=2 * order + 1, axis=0)
    d = 10 * np.log10(np.mean(ltse / noise, axis=1) + 1e-12)
    lo, hi = np.percentile(d, 10), np.percentile(d, 99)
    return smooth_mask(d > lo + frac * (hi - lo), hang=0.05)


def reference_mask(F):
    """Reference speech mask from the quiet ('none') recording: high SNR,
    energy threshold placed close to the noise floor."""
    e = np.convolve(F["energy"], np.ones(3) / 3, mode="same")
    lo, hi = np.percentile(e, 5), np.percentile(e, 99)
    return smooth_mask(e > lo + 0.30 * (hi - lo), min_speech=0.08, min_gap=0.20, hang=0.05)


def align_offset(ref, x, max_lag_s=3.0):
    """Lag (in samples) of x relative to ref found by cross-correlation of the
    band-passed envelopes of the same replayed sentence. Returns (lag, quality)."""
    def env(s):
        s = s - s.mean()
        return np.convolve(np.abs(s), np.ones(80) / 80, mode="same")[::40]   # 400 Hz env
    a, b = env(ref), env(x)
    a, b = (a - a.mean()) / (a.std() + 1e-9), (b - b.mean()) / (b.std() + 1e-9)
    c = correlate(b, a, mode="full", method="fft") / min(len(a), len(b))
    lags = np.arange(-len(a) + 1, len(b))
    w = np.abs(lags) <= int(max_lag_s * SR / 40)
    i = np.argmax(c[w])
    return int(lags[w][i] * 40), float(c[w][i])


def shift_mask(mask, lag_samples, n_frames):
    s = int(round(lag_samples / (HOP * SR)))
    out = np.zeros(n_frames, bool)
    src = np.arange(len(mask))
    dst = src + s
    ok = (dst >= 0) & (dst < n_frames)
    out[dst[ok]] = mask[src[ok]]
    return out


def vad_scores(ref, pred):
    n = min(len(ref), len(pred))
    r, p = ref[:n], pred[:n]
    tp, fp, fn = np.sum(r & p), np.sum(~r & p), np.sum(r & ~p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"acc": float(np.mean(r == p)), "prec": float(prec), "rec": float(rec), "f1": float(f1)}


def try_silero():
    try:
        import torch  # noqa: F401
        model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True,
                                      verbose=False)
        get_ts = utils[0]

        def run(x):
            import torch as _t
            ts = get_ts(_t.from_numpy(x), model, sampling_rate=SR)
            n = 1 + (len(x) - int(FRAME * SR)) // int(HOP * SR)
            m = np.zeros(max(n, 1), bool)
            for t in ts:
                m[t["start"] // int(HOP * SR): t["end"] // int(HOP * SR)] = True
            return m
        return run
    except Exception as e:  # torch missing / no internet - skip quietly
        log(f"  Silero VAD not used ({type(e).__name__}) - optional, needs PyTorch")
        return None


# --------------------------------------------------------------------------
# 4. Models
# --------------------------------------------------------------------------
def build_models():
    return {
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", random_state=SEED)),
        "Neural Network (MLP)": make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                                            max_iter=3000, random_state=SEED)),
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
    }


# --------------------------------------------------------------------------
# 5. Figures
# --------------------------------------------------------------------------
def save(fig, name):
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    log(f"  saved figures/{name}")


def plot_examples(examples):
    fig, axes = plt.subplots(2, 4, figsize=(15, 5.6), gridspec_kw={"height_ratios": [1, 1.4]})
    for j, c in enumerate(CLASSES):
        x, name = examples[c]
        t = np.arange(len(x)) / SR
        axes[0, j].plot(t, x, color=C_CLASS[c], linewidth=0.5)
        axes[0, j].set_title(CLASS_NAMES[c], loc="left")
        axes[0, j].set_xticklabels([])
        f, tt, S = scipy_spectrogram(x, fs=SR, nperseg=400, noverlap=240, nfft=512)
        axes[1, j].pcolormesh(tt, f / 1000, 10 * np.log10(S + 1e-12), shading="auto", cmap="magma")
        axes[1, j].set_xlabel("Time (s)")
        axes[1, j].grid(False)
    axes[0, 0].set_ylabel("Amplitude")
    axes[1, 0].set_ylabel("Frequency (kHz)")
    fig.suptitle("The same kind of speech recording under the four VOiCES background conditions",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig01_background_examples.png")


def plot_dataset(counts, spk_counts):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    vals = [counts[c] for c in CLASSES]
    ax.bar([CLASS_NAMES[c] for c in CLASSES], vals, color=[C_CLASS[c] for c in CLASSES], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v} clips\n{spk_counts[CLASSES[i]]} speakers", ha="center", va="bottom",
                fontsize=8.5, color=INK)
    ax.set_ylabel("Recordings used")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_title("Dataset used for this task (balanced per background)", loc="left")
    ax.grid(axis="x", visible=False)
    save(fig, "fig02_dataset_balance.png")


def plot_model_comparison(cv, chance):
    names = list(cv)
    acc = np.array([cv[n]["acc_mean"] for n in names]) * 100
    sd = np.array([cv[n]["acc_std"] for n in names]) * 100
    order = np.argsort(acc)
    fig, ax = plt.subplots(figsize=(9, 4))
    yy = np.arange(len(names))
    ax.barh(yy, acc[order], xerr=sd[order], height=0.6, color=[PALETTE[i] for i in order],
            error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
    for yi, i in zip(yy, order):
        ax.text(acc[i] + sd[i] + 1, yi, f"{acc[i]:.1f}%", va="center", fontsize=9, color=INK)
    ax.axvline(chance * 100, color=INK_2, linestyle=":", linewidth=1.2)
    ax.text(chance * 100 + 0.8, len(names) - 0.45, f"chance {chance*100:.0f}%", fontsize=8, color=INK_2)
    ax.set_yticks(yy)
    ax.set_yticklabels([names[i] for i in order])
    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy (%) - 5-fold cross-validation grouped by speaker")
    ax.set_title("Background classification: none / babble / music / TV", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "fig03_background_model_comparison.png")


def plot_confusion(cm, title, fname):
    cm = np.array(cm)
    cmn = cm / cm.sum(1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    labels = [CLASS_NAMES[c] for c in CLASSES]
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]*100:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_yticks(range(4))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(False)
    save(fig, fname)


def plot_importance(imp, names):
    order = np.argsort(imp)[::-1][:12][::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(np.arange(len(order)), imp[order], color=PALETTE[0], height=0.6)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=8.5)
    ax.set_xlabel("Random Forest importance")
    ax.set_title("Most useful features for recognising the background", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "fig05_feature_importance.png")


def plot_vad_example(x, ref, preds, title):
    t = np.arange(len(x)) / SR
    tf = np.arange(len(ref)) * HOP
    rows = ["Reference"] + list(preds)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [1.4, 1]})
    a0.plot(t, x, color=INK_2, linewidth=0.4)
    a0.fill_between(tf, -np.abs(x).max(), np.abs(x).max(), where=ref[: len(tf)], color=PALETTE[2],
                    alpha=0.15, linewidth=0, label="Reference speech")
    a0.set_ylabel("Amplitude")
    a0.set_title(title, loc="left")
    a0.legend(loc="upper right")
    for i, r in enumerate(rows):
        m = ref if r == "Reference" else preds[r]
        m = m[: len(tf)]
        a1.fill_between(tf[: len(m)], i + 0.15, i + 0.85, where=m, step="mid",
                        color=PALETTE[2] if r == "Reference" else PALETTE[[0, 1, 6, 3][(i - 1) % 4]],
                        linewidth=0)
    a1.set_yticks(np.arange(len(rows)) + 0.5)
    a1.set_yticklabels(rows)
    a1.set_ylim(0, len(rows))
    a1.invert_yaxis()
    a1.set_xlabel("Time (s)")
    a1.grid(axis="y", visible=False)
    save(fig, "fig06_vad_example.png")


def plot_vad_f1(table, methods):
    xs = np.arange(len(CLASSES))
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    cols = [PALETTE[0], PALETTE[1], PALETTE[6], PALETTE[3]]
    for i, m in enumerate(methods):
        v = [table[m][c]["f1"] * 100 for c in CLASSES]
        ax.bar(xs + (i - (len(methods) - 1) / 2) * w, v, width=w - 0.03, color=cols[i], label=m)
        for xi, vi in zip(xs + (i - (len(methods) - 1) / 2) * w, v):
            ax.text(xi, vi + 1, f"{vi:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([CLASS_NAMES[c] for c in CLASSES])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Speech F1 (%)")
    ax.set_title("Voice Activity Detection - F1 per background condition (test speakers)", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=len(methods))
    ax.grid(axis="x", visible=False)
    save(fig, "fig07_vad_f1_by_background.png")


# --------------------------------------------------------------------------
# 6. README
# --------------------------------------------------------------------------
def write_readme(S):
    cv, names = S["cv"], S["model_names"]
    top = max(cv[n]["acc_mean"] for n in names)
    best = [n for n in names if cv[n]["acc_mean"] >= top - 1e-9]
    rows_cv = "\n".join(
        f"| {n} | {cv[n]['acc_mean']*100:.1f} ± {cv[n]['acc_std']*100:.1f} | {cv[n]['f1_mean']*100:.1f} |"
        for n in names)
    pc = S["per_class_recall"]
    hardest = min(pc, key=pc.get)
    easiest = max(pc, key=pc.get)
    cmb = np.array(S["confusion"])
    off = cmb.astype(float).copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(np.argmax(off), off.shape)
    conf_pair = (CLASS_NAMES[CLASSES[i]], CLASS_NAMES[CLASSES[j]], int(off[i, j]))
    speechy = {"babb", "tele"}
    if {CLASSES[i], CLASSES[j]} <= speechy:
        conf_why = ("babble is made of speech and TV sound contains speech, so both share the "
                    "spectral shape of the foreground voice")
    elif "none" in (CLASSES[i], CLASSES[j]):
        conf_why = ("when the distractor is soft or the microphone is close to the foreground "
                    "loudspeaker, a noisy recording sounds almost like the quiet one")
    else:
        conf_why = ("TV sound mixes speech and music, so it overlaps with both the babble and "
                    "the music conditions")
    imp_top = S["top_features"]
    n_pause = sum("(pauses)" in f for f in imp_top)
    if n_pause >= 4:
        imp_txt = (f"{n_pause} of the 12 most important features are *pause-frame* features, "
                   "confirming that the background is easiest to recognise between the words.")
    else:
        imp_txt = (f"{n_pause} of the 12 most important features are *pause-frame* features; "
                   "most of the information comes from the whole-clip statistics, i.e. the "
                   "background also changes the spectrum while the speaker talks.")

    V, methods = S["vad"], S["vad_methods"]
    vrows = "\n".join(
        f"| {m} | " + " | ".join(f"{V[m][c]['f1']*100:.1f}" for c in CLASSES) +
        f" | **{V[m]['all']['f1']*100:.1f}** |" for m in methods)
    vdetail = "\n".join(
        f"| {m} | {V[m]['all']['acc']*100:.1f} | {V[m]['all']['prec']*100:.1f} | "
        f"{V[m]['all']['rec']*100:.1f} | {V[m]['all']['f1']*100:.1f} |" for m in methods)
    best_vad = max(methods, key=lambda m: V[m]["all"]["f1"])
    d_ltsd = (V["LTSD VAD"]["all"]["f1"] - V["Energy VAD"]["all"]["f1"]) * 100
    d_rf = (V["Learned VAD (RF)"]["all"]["f1"] - V["Energy VAD"]["all"]["f1"]) * 100
    ltsd_txt = (f"Comparing each band with an estimated noise spectrum (LTSD) "
                f"{'improves' if d_ltsd > 0 else 'changes'} the overall F1 by {d_ltsd:+.1f} points, "
                f"and the learned frame classifier by {d_rf:+.1f} points over energy VAD.")
    hard_rf = min(CLASSES, key=lambda c: V["Learned VAD (RF)"][c]["f1"])
    worst_cond = {m: min(CLASSES, key=lambda c: V[m][c]["f1"]) for m in methods}
    e_drop = (V["Energy VAD"]["none"]["f1"] - V["Energy VAD"][worst_cond["Energy VAD"]]["f1"]) * 100
    al = S["alignment"]
    if S.get("ref_mode") == "aligned":
        ref_txt = (f"3. Pairs with a weak correlation peak are discarded.\n"
                   f"   Aligned noisy recordings: **{al['ok']} of {al['tried']}** that had a quiet twin "
                   f"(median peak correlation {al['median_q']:.2f}). Noisy recordings without an "
                   f"aligned quiet twin are left out of the VAD evaluation.")
    else:
        ref_txt = (f"3. **Limitation of this run:** only {al['ok']} noisy recordings could be aligned "
                   f"with a quiet twin, so the reference of every recording was taken from a "
                   f"conservative energy mask of the same file. The VAD scores below therefore "
                   f"measure agreement with that reference and favour the energy-based detector; "
                   f"they should be read as indicative only.")
    seg_cnt = S["counts"]

    txt = f"""# Mini-Task 9 — Background Classification & Voice Activity Detection (VOiCES)

**Course:** Digital Speech Processing — Dr. Khalooei
**Student:** Ghassan Alsafar — **ID:** 404131085

---

## 1. Goal

Using the **VOiCES** corpus (*Voices Obscured in Complex Environmental Settings*):

* **Part A — Events & background classification:** recognise which background
  sound is playing in a room recording — **none, babble, music or TV**.
* **Part B — Speech / non-speech & VAD:** decide, every 10 ms, whether a person is
  speaking, and measure how each background makes this harder.

## 2. Dataset — VOiCES

VOiCES was recorded by SRI and Lab41 (IQT Labs): LibriSpeech sentences were
**replayed through a loudspeaker in real rooms** and recorded by several
microphones at different distances, while a second loudspeaker played a
**distractor** sound. The full corpus is 27–417 GB, so this task uses the
**official subset** published by the authors on GitHub (`Lab41/VOiCES-subset`,
foreground loudspeaker at 90°). The script downloads only as many subset archives
as needed.

All labels come from the file name, e.g.
`Lab41-SRI-VOiCES-rm1-babb-sp0083-ch003054-sg0005-mc01-stu-clo-dg090.wav` →
room 1, **babble**, speaker 0083, chapter, segment, mic 01 (studio mic, close), 90°.

| Property | Value |
|---|---|
| Recordings available after download | {S['n_available']} |
| Recordings used (balanced) | {S['n_used']} ({", ".join(f"{CLASS_NAMES[c]} {seg_cnt[c]}" for c in CLASSES)}) |
| Speakers | {S['n_speakers']} |
| Rooms / microphones | {", ".join(S['rooms'])} / {S['n_mics']} microphones |
| Sample rate | {SR} Hz mono (resampled if needed), first {S['max_sec']:.0f} s of each file |

![Background examples](figures/fig01_background_examples.png)

![Dataset](figures/fig02_dataset_balance.png)

## 3. Part A — Background classification

### Method

* **Frame features** (25 ms, 10 ms hop, implemented with NumPy): log energy,
  13 MFCC, spectral centroid, spectral flatness, roll-off, zero-crossing rate.
* **Clip descriptor ({S['n_feat']} values):** mean and std of all frame features,
  std of ΔMFCC, energy percentiles, **and the mean/std of the features measured
  only in the quietest 30 % of frames.** Those frames are the pauses between words,
  where the background is heard on its own — this is the key idea that makes the
  background visible behind the foreground speech.
* **Six classifiers** (same family as Mini-Task 8, incl. SVM and a Neural Network).
* **Evaluation:** 5-fold cross-validation **grouped by speaker** — a test speaker is
  never seen in training, so the model must recognise the *background*, not the voice.

### Results

| Model | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
{rows_cv}

Chance level for 4 balanced classes is **25 %**. Best: {", ".join(f"**{n}**" for n in best)}
— {top*100:.1f} %.

![Model comparison](figures/fig03_background_model_comparison.png)

![Confusion matrix](figures/fig04_background_confusion.png)

![Feature importance](figures/fig05_feature_importance.png)

**Observations.** The easiest condition is **{CLASS_NAMES[easiest]}**
(recall {pc[easiest]*100:.0f} %) and the hardest is **{CLASS_NAMES[hardest]}**
(recall {pc[hardest]*100:.0f} %). The most frequent confusion is
**{conf_pair[0]} → {conf_pair[1]}** ({conf_pair[2]} clips): {conf_why}.
{imp_txt}

## 4. Part B — Speech / non-speech & Voice Activity Detection

### Reference labels (ground truth)

VOiCES has no frame-level speech labels, so they were built from the data itself:

1. For every sentence, the recording in the **quiet condition (none)** has a high
   signal-to-noise ratio; an energy threshold close to its noise floor gives a
   reliable speech mask.
2. The **same LibriSpeech sentence, same microphone** was replayed in the babble,
   music and TV sessions. The noisy recording is aligned to the quiet one by
   **cross-correlation of their envelopes** and the mask is shifted by the found lag.
{ref_txt}

### VAD methods compared

| Method | Idea |
|---|---|
| Energy VAD | adaptive threshold between noise floor and speech peak, gap filling, hangover |
| LTSD VAD | Long-Term Spectral Divergence (Ramírez et al., 2004): compares each frame's long-term spectral envelope with a noise spectrum estimated from the quietest frames — a classic statistical, noise-adaptive VAD |
| Learned VAD (RF) | Random Forest on per-frame MFCC/energy/flatness with ±50 ms context, trained on the *training speakers* in all four backgrounds |
{"| Silero VAD | pre-trained neural VAD (PyTorch) |" if "Silero VAD" in methods else ""}

Evaluation on **held-out test speakers** ({S['vad_test_files']} recordings,
{S['vad_frames']:,} frames of 10 ms).

### Results — speech F1 (%) per background

| Method | {" | ".join(CLASS_NAMES[c] for c in CLASSES)} | All |
|---|{"|".join(["---:"] * (len(CLASSES) + 1))}|
{vrows}

| Method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
{vdetail}

![VAD example](figures/fig06_vad_example.png)

![VAD F1](figures/fig07_vad_f1_by_background.png)

**Observations.** The best detector overall is **{best_vad}**
(F1 {V[best_vad]['all']['f1']*100:.1f} %). The simple energy VAD loses
**{e_drop:.1f} F1 points** between the quiet condition and its hardest condition
({CLASS_NAMES[worst_cond['Energy VAD']]}): the distractor raises the noise floor and
fills the pauses, so a pure loudness rule cannot tell speech from background.
{ltsd_txt} A classifier that has *learned* what speech frames look like in every
background uses many cues at once (formant structure in the MFCCs, flatness,
energy dynamics); its hardest condition is still
**{CLASS_NAMES[hard_rf]}**, where the background itself can contain speech-like sound.

## 5. Conclusions

1. The background of a far-field recording can be recognised well above chance
   from simple spectral statistics, especially when they are measured in the pauses
   between words.
2. Each background affects the two tasks differently — see the per-condition
   tables above for which one is hardest here.
3. Energy-only VAD is fine in quiet rooms but loses accuracy in complex
   environments; noise-adaptive (LTSD) and learned multi-feature VADs are more robust.

**Limitations.** Only the 90° subset and the first {S['max_sec']:.0f} s of each
file were used; reference labels are derived from the quiet recording (not hand
annotated). Next steps: more data from the full devkit, CNN on log-mel
spectrograms, and comparing with neural VADs such as Silero or WebRTC.

## 6. How to run

```bash
pip install -r requirements.txt
python voices_background_vad.py
```

Or open `MiniTask09_VOiCES.ipynb` in Jupyter and run all cells. The script
downloads the subset archives it needs (about 100 MB each) into `data/`, then
writes `figures/`, `results/results.json` and this README.

## 7. Files

```
MiniTask09_VOiCES/
├── voices_background_vad.py   # the whole pipeline
├── MiniTask09_VOiCES.ipynb    # notebook that runs it and shows the results
├── requirements.txt
├── README.md                  # this report (auto-generated)
├── figures/                   # 7 figures
└── results/results.json       # all numbers
```

*Environment: Python {S['env']['python']}, scikit-learn {S['env']['sklearn']},
NumPy {S['env']['numpy']}. Run time {S['runtime']:.0f} s.*

**Reference:** C. Richey et al., "Voices Obscured in Complex Environmental Settings
(VOiCES) corpus", Interspeech 2018.
"""
    (HERE / "README.md").write_text(txt, encoding="utf-8")
    log("  saved README.md")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--per-class", type=int, default=60, help="recordings used per background")
    ap.add_argument("--max-tars", type=int, default=8, help="max subset archives to download")
    ap.add_argument("--min-download", type=int, default=30,
                    help="stop downloading once every background has this many recordings")
    ap.add_argument("--max-sec", type=float, default=12.0, help="seconds used from each file")
    ap.add_argument("--no-silero", action="store_true")
    args = ap.parse_args()
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    log("[1/6] VOiCES data")
    root = ensure_dataset(args.data_dir, min(args.min_download, args.per_class), args.max_tars)
    files = wav_files(root)
    cnt = class_counts(files)
    log(f"  {len(files)} recordings available: " + ", ".join(f"{k}={cnt.get(k, 0)}" for k in CLASSES))
    missing = [c for c in CLASSES if cnt.get(c, 0) == 0]
    if missing:
        sys.exit(f"No recordings for {missing}. Try --max-tars 8.")
    per_class = min(args.per_class, min(cnt[c] for c in CLASSES))
    chosen = pick_files(files, per_class, rng)
    log(f"  using {per_class} recordings per background -> {len(chosen)} recordings")

    log("[2/6] Features")
    recs = []
    for p, d in chosen:
        x = load_wav(p, args.max_sec)
        if len(x) < SR:
            continue
        F = frame_features(x)
        recs.append({"path": p, "meta": d, "x": x, "F": F, "vec": clip_vector(F)})
    X = np.array([r["vec"] for r in recs])
    y = np.array([CLASSES.index(r["meta"]["dist"]) for r in recs])
    spk = np.array([r["meta"]["spk"] for r in recs])
    log(f"  {len(recs)} clips, {X.shape[1]} features, {len(set(spk))} speakers")
    examples = {}
    for c in CLASSES:
        r = next(r for r in recs if r["meta"]["dist"] == c)
        examples[c] = (r["x"][: 6 * SR], r["path"].name)
    plot_examples(examples)
    counts = Counter(r["meta"]["dist"] for r in recs)
    spk_counts = {c: len({r["meta"]["spk"] for r in recs if r["meta"]["dist"] == c}) for c in CLASSES}
    plot_dataset(counts, spk_counts)

    log("[3/6] Part A - background classification (5-fold CV grouped by speaker)")
    names = list(build_models())
    n_splits = min(5, len(set(spk)))
    cv, all_pred = {}, {}
    for n in names:
        accs, f1s, pred = [], [], np.zeros_like(y)
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, spk):
            m = build_models()[n].fit(X[tr], y[tr])
            p = m.predict(X[te])
            pred[te] = p
            accs.append(accuracy_score(y[te], p))
            f1s.append(f1_score(y[te], p, average="macro"))
        cv[n] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                 "f1_mean": float(np.mean(f1s)), "fold_acc": [float(a) for a in accs]}
        all_pred[n] = pred
        log(f"  {n:<22} {cv[n]['acc_mean']*100:6.1f}% ± {cv[n]['acc_std']*100:.1f}")
    plot_model_comparison(cv, 1 / len(CLASSES))
    best = max(names, key=lambda n: cv[n]["acc_mean"])
    cm = confusion_matrix(y, all_pred[best], labels=range(4))
    plot_confusion(cm, f"Confusion matrix - {best} (all CV folds)", "fig04_background_confusion.png")
    per_class_recall = {c: float(cm[i, i] / max(cm[i].sum(), 1)) for i, c in enumerate(CLASSES)}
    base = ["logE"] + [f"MFCC{i}" for i in range(13)] + ["centroid", "flatness", "rolloff", "ZCR"]
    feat_names = ([f"{b} mean" for b in base[1:]] + [f"{b} std" for b in base[1:]] +
                  [f"dMFCC{i} std" for i in range(13)] + [f"{b} mean (pauses)" for b in base[1:]] +
                  [f"{b} std (pauses)" for b in base[1:]] + ["energy p10", "energy p90", "energy range"])
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED).fit(X, y)
    plot_importance(rf.feature_importances_, feat_names)
    top_features = [feat_names[i] for i in np.argsort(rf.feature_importances_)[::-1][:12]]

    log("[4/6] Part B - reference speech labels (quiet recording + alignment)")
    # quiet ("none") recordings on disk, indexed by sentence (room + speaker/chapter/segment)
    # and by sentence alone (any room); the same mic is preferred when it exists
    quiet_by_key, quiet_by_sent = defaultdict(list), defaultdict(list)
    for fp in files:
        d = parse_name(fp.stem)
        if d["dist"] == "none":
            quiet_by_key[d["key"]].append((fp, d))
            quiet_by_sent[d["sentence"]].append((fp, d))
    loaded = {}

    def quiet_twin(meta):
        cands = quiet_by_key.get(meta["key"]) or quiet_by_sent.get(meta["sentence"]) or []
        if not cands:
            return None
        fp, _ = sorted(cands, key=lambda t: (t[1]["mic"] != meta["mic"], t[0].name))[0]
        if fp not in loaded:
            qx = load_wav(fp, args.max_sec + 3)
            loaded[fp] = {"x": qx, "F": frame_features(qx)}
        return loaded[fp]

    tried, ok_n, quals = 0, 0, []
    for r in recs:
        dist = r["meta"]["dist"]
        n_fr = len(r["F"]["energy"])
        if dist == "none":
            r["ref"] = reference_mask(r["F"])
            continue
        q_rec = quiet_twin(r["meta"])
        if q_rec is None:
            continue
        tried += 1
        lag, qual = align_offset(q_rec["x"], r["x"])
        quals.append(qual)
        if qual < 0.35:
            continue
        ok_n += 1
        r["ref"] = shift_mask(reference_mask(q_rec["F"]), lag, n_fr)
    with_ref = [r for r in recs if "ref" in r]
    log(f"  aligned {ok_n}/{tried} noisy recordings; {len(with_ref)} recordings have reference labels")
    ref_mode = "aligned"
    if ok_n < 10:
        ref_mode = "self"
        log("  (too few quiet twins - using each file's own conservative energy mask as reference;"
            " the README states this limitation)")
        for r in recs:
            r["ref"] = reference_mask(r["F"])
        with_ref = recs

    log("[5/6] Part B - VAD comparison on held-out speakers")
    spks = sorted({r["meta"]["spk"] for r in with_ref})
    rng.shuffle(spks)
    test_spk = set(spks[: max(1, int(round(len(spks) * 0.3)))])
    train = [r for r in with_ref if r["meta"]["spk"] not in test_spk]
    test = [r for r in with_ref if r["meta"]["spk"] in test_spk]
    if not train:
        train, test = with_ref[::2], with_ref[1::2]
    Xf = np.vstack([frame_matrix(r["F"]) for r in train])
    yf = np.concatenate([r["ref"][: len(r["F"]["energy"])] for r in train])
    if len(yf) > 80000:
        idx = rng.choice(len(yf), 80000, replace=False)
        Xf, yf = Xf[idx], yf[idx]
    frame_clf = RandomForestClassifier(n_estimators=120, min_samples_leaf=5, n_jobs=-1,
                                       random_state=SEED).fit(Xf, yf)
    silero = None if args.no_silero else try_silero()
    methods = ["Energy VAD", "LTSD VAD", "Learned VAD (RF)"] + (["Silero VAD"] if silero else [])
    pooled = {m: {c: ([], []) for c in CLASSES + ["all"]} for m in methods}
    example = None
    for r in test:
        F, ref = r["F"], r["ref"]
        preds = {
            "Energy VAD": energy_vad(F),
            "LTSD VAD": ltsd_vad(F),
            "Learned VAD (RF)": smooth_mask(frame_clf.predict_proba(frame_matrix(F))[:, 1] > 0.5,
                                            hang=0.05),
        }
        if silero:
            preds["Silero VAD"] = silero(r["x"])
        n = len(ref)
        for m, p in preds.items():
            p = np.pad(p[:n], (0, max(0, n - len(p))))
            for c in (r["meta"]["dist"], "all"):
                pooled[m][c][0].append(ref)
                pooled[m][c][1].append(p)
        if example is None and r["meta"]["dist"] == "babb":
            example = (r, preds)
    vad = {m: {c: vad_scores(np.concatenate(v[0]), np.concatenate(v[1])) if v[0] else
               {"acc": 0, "prec": 0, "rec": 0, "f1": 0}
               for c, v in pooled[m].items()} for m in methods}
    for m in methods:
        log(f"  {m:<18} F1: " + ", ".join(f"{c} {vad[m][c]['f1']*100:.0f}%" for c in CLASSES) +
            f"  | all {vad[m]['all']['f1']*100:.1f}%")
    if example is None:
        example = (test[0], None)
    er = example[0]
    ep = example[1] or {"Energy VAD": energy_vad(er["F"]), "LTSD VAD": ltsd_vad(er["F"])}
    plot_vad_example(er["x"], er["ref"], ep,
                     f"VAD on a {CLASS_NAMES[er['meta']['dist']].lower()} recording (test speaker "
                     f"{er['meta']['spk']})")
    plot_vad_f1(vad, methods)

    log("[6/6] Report")
    summary = {
        "n_available": len(files), "n_used": len(recs), "counts": dict(counts),
        "n_speakers": len(set(spk)), "rooms": sorted({r["meta"]["room"] for r in recs}),
        "n_mics": len({r["meta"]["mic"] for r in recs}), "max_sec": args.max_sec,
        "n_feat": int(X.shape[1]), "model_names": names, "cv": cv,
        "confusion": cm.tolist(), "best_model": best, "per_class_recall": per_class_recall,
        "top_features": top_features,
        "ref_mode": ref_mode,
        "alignment": {"tried": tried, "ok": ok_n,
                      "median_q": float(np.median(quals)) if quals else 0.0},
        "vad_methods": methods, "vad": vad, "vad_test_files": len(test),
        "vad_frames": int(sum(len(r["ref"]) for r in test)),
        "test_speakers": sorted(test_spk),
        "env": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                "numpy": np.__version__},
    }
    summary["runtime"] = time.time() - t0
    (RES_DIR / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(summary)
    log(f"\nDone in {time.time() - t0:.0f} s. Open README.md to see the report.")


if __name__ == "__main__":
    main()
