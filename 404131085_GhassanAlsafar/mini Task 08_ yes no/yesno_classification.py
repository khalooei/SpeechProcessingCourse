"""
Mini-Task 8 - Part 1: Spoken "Yes" / "No" Word Classification
==============================================================

Course   : Digital Speech Processing - Dr. Khalooei
Student  : Ghassan Alsafar (404131085)

What this script does (end to end, one command):
  1. Downloads the OpenSLR "yesno" dataset (60 recordings, 8 words each).
  2. Explores the data: signal, sample rate, labels, waveform and spectrogram.
  3. Splits every recording into its 8 words with an adaptive energy detector.
  4. Extracts MFCC + delta-MFCC features (implemented from scratch with NumPy).
  5. Trains and compares 6 classifiers, including SVM and a Neural Network (MLP),
     with recording-level (grouped) cross-validation so no speaker/recording
     leaks between train and test.
  6. Stress-tests every model with added noise (clean -> -5 dB SNR) and compares
     clean training with multi-condition (noise-augmented) training.
  7. Saves all figures to ./figures, all numbers to ./results and writes README.md.

Run:
    pip install -r requirements.txt
    python yesno_classification.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import spectrogram as scipy_spectrogram

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sklearn  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import GroupKFold, GroupShuffleSplit  # noqa: E402
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
OWN_DIR = HERE / "my_voice"          # Part B: my own recordings

DATA_URLS = [
    "https://www.openslr.org/resources/1/waves_yesno.tar.gz",
    "https://us.openslr.org/resources/1/waves_yesno.tar.gz",
    "https://openslr.elda.org/resources/1/waves_yesno.tar.gz",
]

WORDS_PER_FILE = 8
CLASS_NAMES = ["NO", "YES"]          # label 0 = "no" (lo), 1 = "yes" (ken)
SEED = 42
SNR_LEVELS = [None, 20, 15, 10, 5, 0, -5]   # None = clean
NOISE_REPEATS = 5

# Chart style (validated categorical palette, fixed order)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
           "#6250d6", "#e34948"]
C_YES, C_NO = PALETTE[0], PALETTE[1]
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
def ensure_dataset(data_dir: Path) -> Path:
    """Download + extract the yesno dataset if it is not already present."""
    wav_dir = data_dir / "waves_yesno"
    if wav_dir.is_dir() and len(list(wav_dir.glob("*.wav"))) >= 60:
        return wav_dir
    if any(data_dir.glob("*.wav")) if data_dir.is_dir() else False:
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "waves_yesno.tar.gz"
    if not archive.exists():
        last_err = None
        for url in DATA_URLS:
            try:
                log(f"  downloading {url} ...")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(archive, "wb") as f:
                    f.write(r.read())
                break
            except Exception as e:  # try the next mirror
                last_err = e
                if archive.exists():
                    archive.unlink()
        else:
            sys.exit(
                "\nCould not download the dataset automatically "
                f"({last_err}).\nDownload it by hand from\n  {DATA_URLS[0]}\n"
                f"and put the file 'waves_yesno.tar.gz' inside:\n  {data_dir}\n"
                "then run this script again."
            )
    log("  extracting ...")
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers()
                   if m.isfile() and not m.name.startswith(("/", ".."))]
        tar.extractall(data_dir, members=members)
    return wav_dir


def load_dataset(wav_dir: Path):
    files = sorted(p for p in wav_dir.glob("*.wav") if p.stem.count("_") == WORDS_PER_FILE - 1)
    recs = []
    for p in files:
        sr, x = wavfile.read(p)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.float32) / np.iinfo(x.dtype).max
        labels = [int(t) for t in p.stem.split("_")]
        recs.append({"name": p.stem, "sr": int(sr), "x": x.astype(np.float32), "labels": labels})
    if not recs:
        sys.exit(f"No yesno .wav files found in {wav_dir}")
    return recs


# --------------------------------------------------------------------------
# 2. Signal processing helpers
# --------------------------------------------------------------------------
def frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n = 1 + (len(x) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def frame_energy_db(x, sr, win=0.025, hop=0.010):
    fl, hp = int(win * sr), int(hop * sr)
    frames = frame_signal(x, fl, hp)
    e = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
    e = np.convolve(e, np.ones(5) / 5, mode="same")      # light smoothing
    return e, hp


def _runs(mask: np.ndarray):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def segment_words(x, sr, n_expected=WORDS_PER_FILE, min_gap=0.15, min_len=0.10, pad=0.03,
                  hang=0.15):
    """Adaptive energy-based word segmentation.

    The threshold is placed between the noise floor (10th percentile of frame
    energy) and the speech peak (99th percentile). Several threshold levels are
    tried and the one that gives exactly `n_expected` words is kept. Returns
    (segments in samples, energy curve, chosen threshold, hop).
    """
    e, hop = frame_energy_db(x, sr)
    floor, peak = np.percentile(e, 10), np.percentile(e, 99)
    best = None
    for frac in np.linspace(0.15, 0.70, 23):
        thr = floor + frac * (peak - floor)
        segs = _runs(e > thr)
        merged = []
        for s, t in segs:                                   # close short gaps
            if merged and (s - merged[-1][1]) * hop / sr < min_gap:
                merged[-1] = (merged[-1][0], t)
            else:
                merged.append((s, t))
        merged = [(s, t) for s, t in merged if (t - s) * hop / sr >= min_len]
        score = abs(len(merged) - n_expected)
        if best is None or score < best[0]:
            best = (score, merged, thr)
        if score == 0:
            break
    _, segs, thr = best
    p = int(pad * sr)
    # hangover: keep up to `hang` s after each word so weak endings (the /s/ of "yes")
    # are not cut off, but never cross the middle of the pause to the next word
    out = []
    for i, (s, t) in enumerate(segs):
        start = max(0, s * hop - p)
        end = t * hop + int(0.025 * sr) + int(hang * sr)
        if i + 1 < len(segs):
            end = min(end, (t * hop + segs[i + 1][0] * hop) // 2)
        out.append((start, min(len(x), end)))
    return out, e, thr, hop


def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)


_FB_CACHE: dict = {}


def mel_filterbank(sr, n_fft, n_mels=26):
    key = (sr, n_fft, n_mels)
    if key in _FB_CACHE:
        return _FB_CACHE[key]
    mels = np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(r - c, 1)
    _FB_CACHE[key] = fb
    return fb


def mfcc(x, sr, n_mfcc=13, n_mels=26, win=0.025, hop=0.010, n_fft=None):
    """MFCC from scratch: pre-emphasis -> framing -> Hamming -> |FFT|^2 -> Mel -> log -> DCT."""
    if n_fft is None:                                   # 512 at 8/16 kHz, larger at 44.1/48 kHz
        n_fft = max(512, 1 << int(np.ceil(np.log2(int(win * sr)))))
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    frames = frame_signal(x, int(win * sr), int(hop * sr)) * np.hamming(int(win * sr))
    pow_spec = (np.abs(np.fft.rfft(frames, n_fft)) ** 2) / n_fft
    mel_e = pow_spec @ mel_filterbank(sr, n_fft, n_mels).T
    log_mel = np.log(mel_e + 1e-10)
    return dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]


def word_features(w, sr):
    """Fixed-length vector per word: mean+std of MFCC and delta-MFCC (52 values)."""
    m = mfcc(w, sr)
    d = np.gradient(m, axis=0) if len(m) > 1 else np.zeros_like(m)
    return np.concatenate([m.mean(0), m.std(0), d.mean(0), d.std(0)])


def add_noise(x, snr_db, rng):
    if snr_db is None:
        return x
    p_sig = np.mean(x ** 2) + 1e-12
    noise = rng.standard_normal(len(x)).astype(np.float32)
    noise *= np.sqrt(p_sig / (10 ** (snr_db / 10)) / np.mean(noise ** 2))
    return x + noise


# --------------------------------------------------------------------------
# 3. Models
# --------------------------------------------------------------------------
def build_models():
    return {
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", random_state=SEED)),
        "Neural Network (MLP)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=2000,
                          random_state=SEED)),
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
    }



def segment_recordings(recs):
    """Split every recording into its words; recordings without exactly 8 words are skipped."""
    words, srs, y, groups, failed, cache = [], [], [], [], [], {}
    for gi, r in enumerate(recs):
        segs, e, thr, hop = segment_words(r["x"], r["sr"])
        cache[r["name"]] = (segs, e, thr, hop)
        if len(segs) != WORDS_PER_FILE:
            failed.append(r["name"])
            continue
        for (s, en), lab in zip(segs, r["labels"]):
            words.append(r["x"][s:en])
            srs.append(r["sr"])
            y.append(lab)
            groups.append(gi)
    return words, srs, np.array(y), np.array(groups), failed, cache


def grouped_cv(X, y, groups, names, n_splits=5):
    cv = {}
    n_splits = min(n_splits, len(set(groups)))
    for n in names:
        accs, f1s, fit = [], [], []
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
            m = build_models()[n]
            t = time.perf_counter()
            m.fit(X[tr], y[tr])
            fit.append((time.perf_counter() - t) * 1000)
            p = m.predict(X[te])
            accs.append(accuracy_score(y[te], p))
            f1s.append(f1_score(y[te], p, average="macro"))
        cv[n] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                 "f1_mean": float(np.mean(f1s)), "fit_ms": float(np.mean(fit)),
                 "fold_acc": [float(a) for a in accs], "n_splits": n_splits}
        log(f"  {n:<22} {cv[n]['acc_mean']*100:6.2f}% ± {cv[n]['acc_std']*100:.2f}")
    return cv


def plot_dataset_vs_own(cv_ds, cv_own, names):
    yy = np.arange(len(names))
    h = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    a_ds = np.array([cv_ds[n]["acc_mean"] for n in names]) * 100
    a_ow = np.array([cv_own[n]["acc_mean"] for n in names]) * 100
    ax.barh(yy - h / 2, a_ds, height=h - 0.04, color=PALETTE[0], label="Course dataset (yesno)")
    ax.barh(yy + h / 2, a_ow, height=h - 0.04, color=PALETTE[1], label="My own voice")
    lo = max(0, min(80, np.floor(min(a_ds.min(), a_ow.min()) / 5) * 5 - 5))
    for yi, v in zip(yy - h / 2, a_ds):
        ax.text(v - (100 - lo) * 0.012, yi, f"{v:.1f}%", va="center", ha="right", fontsize=8.5,
                color="white", fontweight="bold")
    for yi, v in zip(yy + h / 2, a_ow):
        ax.text(v - (100 - lo) * 0.012, yi, f"{v:.1f}%", va="center", ha="right", fontsize=8.5,
                color="white", fontweight="bold")
    ax.set_yticks(yy)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(lo, 100.5)
    ax.set_xlabel("Accuracy (%) - cross-validation grouped by recording")
    ax.set_title("Course dataset vs. my own voice", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    save(fig, "fig10_dataset_vs_my_voice.png")


def find_own_voice(path):
    """My recordings live in ./my_voice; if the .wav files were uploaded loose, look next to the script."""
    for d in [path, HERE]:
        if d is not None and d.is_dir() and any(p.stem.count("_") == WORDS_PER_FILE - 1
                                                for p in d.glob("*.wav")):
            return d
    return None

# --------------------------------------------------------------------------
# 4. Figures
# --------------------------------------------------------------------------
def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    log(f"  saved figures/{name}")


def plot_overview(rec, segs, fname="fig01_signal_and_spectrogram.png", who="Recording"):
    x, sr = rec["x"], rec["sr"]
    t = np.arange(len(x)) / sr
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.2]})
    a1.plot(t, x, color=INK_2, linewidth=0.6)
    top = float(np.abs(x).max())
    a1.set_ylim(-top * 1.1, top * 1.45)
    for (s, e), lab in zip(segs, rec["labels"]):
        c = C_YES if lab else C_NO
        a1.axvspan(s / sr, e / sr, color=c, alpha=0.18, linewidth=0)
        a1.text((s + e) / 2 / sr, top * 1.3, CLASS_NAMES[lab], ha="center", va="center",
                fontsize=9, fontweight="bold", color=INK)
    a1.set_ylabel("Amplitude")
    a1.set_title(f"{who} '{rec['name']}.wav'  |  sample rate = {sr} Hz  |  "
                 f"duration = {len(x)/sr:.2f} s  |  labels = {rec['labels']}", loc="left")
    f, tt, S = scipy_spectrogram(x, fs=sr, nperseg=int(0.025 * sr), noverlap=int(0.015 * sr), nfft=512)
    a2.pcolormesh(tt, f, 10 * np.log10(S + 1e-10), shading="auto", cmap="magma")
    a2.set_ylabel("Frequency (Hz)")
    a2.set_xlabel("Time (s)")
    a2.set_title("Spectrogram (25 ms Hann window, 10 ms hop)", loc="left")
    a2.grid(False)
    save(fig, fname)


def plot_segmentation(rec, segs, e, thr, hop):
    sr = rec["sr"]
    t = np.arange(len(e)) * hop / sr
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(t, e, color=PALETTE[6], label="Short-time energy (dB)")
    ax.axhline(thr, color=INK_2, linestyle="--", linewidth=1.2, label=f"Adaptive threshold ({thr:.1f} dB)")
    for i, (s, en) in enumerate(segs):
        ax.axvspan(s / sr, en / sr, color=PALETTE[2], alpha=0.18, linewidth=0,
                   label="Detected word" if i == 0 else None)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (dB)")
    ax.set_title(f"Energy-based word segmentation - {len(segs)} words found in '{rec['name']}.wav'", loc="left")
    ax.legend(loc="lower right", ncol=3)
    save(fig, "fig02_energy_segmentation.png")


def plot_class_examples(words_yes, words_no, sr, X, y, fname="fig03_yes_vs_no.png", who=""):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for ax, w, name in [(axes[0], words_yes, "YES"), (axes[1], words_no, "NO")]:
        f, tt, S = scipy_spectrogram(w, fs=sr, nperseg=int(0.025 * sr), noverlap=int(0.015 * sr), nfft=512)
        ax.pcolormesh(tt, f, 10 * np.log10(S + 1e-10), shading="auto", cmap="magma")
        ax.set_title(f'{who}Example word: "{name}"', loc="left")
        ax.set_xlabel("Time (s)")
        ax.grid(False)
    axes[0].set_ylabel("Frequency (Hz)")
    k = np.arange(1, 13)                     # c0 (log energy) left out - it dwarfs the rest
    for lab, c in [(1, C_YES), (0, C_NO)]:
        mu, sd = X[y == lab, 1:13].mean(0), X[y == lab, 1:13].std(0)
        axes[2].plot(k, mu, color=c, marker="o", markersize=5, label=CLASS_NAMES[lab])
        axes[2].fill_between(k, mu - sd, mu + sd, color=c, alpha=0.15, linewidth=0)
    axes[2].set_title("Average MFCC 1-12 per class (±1 std)", loc="left")
    axes[2].set_xlabel("MFCC coefficient")
    axes[2].set_xticks(k)
    axes[2].legend()
    save(fig, fname)


def plot_pca(X, y):
    Z = PCA(n_components=2, random_state=SEED).fit_transform(StandardScaler().fit_transform(X))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for lab, c in [(1, C_YES), (0, C_NO)]:
        ax.scatter(Z[y == lab, 0], Z[y == lab, 1], s=28, color=c, edgecolor="white",
                   linewidth=0.8, alpha=0.85, label=CLASS_NAMES[lab])
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title("Feature space (PCA of 52 MFCC features)", loc="left")
    ax.legend()
    save(fig, "fig04_feature_space_pca.png")


def plot_model_comparison(cv):
    names = list(cv)
    acc = np.array([cv[n]["acc_mean"] for n in names]) * 100
    sd = np.array([cv[n]["acc_std"] for n in names]) * 100
    order = np.argsort(acc)
    fig, ax = plt.subplots(figsize=(9, 4))
    yy = np.arange(len(names))
    ax.barh(yy, acc[order], xerr=sd[order], height=0.6,
            color=[PALETTE[names.index(names[i])] for i in order],
            error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
    lo = max(0, min(90, np.floor((acc - sd).min() / 5) * 5 - 5))
    for yi, i in zip(yy, order):
        ax.text(acc[i] - (100 - lo) * 0.015, yi, f"{acc[i]:.2f}%", va="center", ha="right",
                fontsize=9, color="white", fontweight="bold")
    ax.set_yticks(yy)
    ax.set_yticklabels([names[i] for i in order])
    ax.set_xlim(lo, 100.5)
    ax.set_xlabel("Accuracy (%) - 5-fold cross-validation grouped by recording")
    ax.set_title("Classifier comparison on clean speech", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "fig05_model_comparison.png")


def plot_confusions(cms, names):
    n = len(names)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.1 * rows))
    for ax, name in zip(np.ravel(axes), names):
        cm = np.array(cms[name])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(1).max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                        color="white" if cm[i, j] > cm.sum(1).max() / 2 else INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_title(name, fontsize=10, loc="left")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.grid(False)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices - noise-augmented models on held-out recordings at 5 dB SNR",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig06_confusion_matrices_5dB.png")


def plot_noise(noise, names):
    labels = ["clean" if s is None else f"{s}" for s in SNR_LEVELS]
    xs = np.arange(len(SNR_LEVELS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
    for ax, mode, title in [(axes[0], "clean_train", "Trained on clean speech"),
                            (axes[1], "multi_train", "Trained with noise augmentation")]:
        for i, n in enumerate(names):
            ax.plot(xs, np.array(noise[mode][n]) * 100, color=PALETTE[i], marker="o",
                    markersize=5, label=n)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Test SNR (dB)  - noisier to the right")
        ax.set_title(title, loc="left")
        ax.axhline(50, color=INK_2, linestyle=":", linewidth=1)
        ax.text(xs[0], 48.5, "chance level (50%)", ha="left", va="top", fontsize=8, color=INK_2)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_ylim(40, 103)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(names), bbox_to_anchor=(0.5, -0.12), fontsize=9)
    save(fig, "fig07_noise_robustness.png")


# --------------------------------------------------------------------------
# 5. README
# --------------------------------------------------------------------------
def write_readme(S):
    cv, names = S["cv"], S["model_names"]
    top = max(cv[n]["acc_mean"] for n in names)
    best_clean = [n for n in names if cv[n]["acc_mean"] >= top - 1e-9]
    best_clean_txt = (", ".join(f"**{n}**" for n in best_clean) +
                      (" (tied)" if len(best_clean) > 1 else "") + f" — {top*100:.2f}%")
    i0 = SNR_LEVELS.index(0)
    # robustness = mean accuracy over all noisy conditions (area under the SNR curve)
    rob = {n: float(np.mean(S["noise"]["clean_train"][n][1:])) for n in names}
    rob_m = {n: float(np.mean(S["noise"]["multi_train"][n][1:])) for n in names}
    best_rob, worst_rob = max(rob, key=rob.get), min(rob, key=rob.get)
    best_rob_m = max(rob_m, key=rob_m.get)
    gain = np.mean([rob_m[n] - rob[n] for n in names]) * 100
    a0 = {n: S["noise"]["clean_train"][n][i0] for n in names}
    a0m = {n: S["noise"]["multi_train"][n][i0] for n in names}
    rows_cv = "\n".join(
        f"| {n} | {cv[n]['acc_mean']*100:.2f} ± {cv[n]['acc_std']*100:.2f} | "
        f"{cv[n]['f1_mean']*100:.2f} | {cv[n]['fit_ms']:.1f} |" for n in names)
    snr_hdr = " | ".join("clean" if s is None else f"{s} dB" for s in SNR_LEVELS)
    snr_sep = "|".join(["---:"] * len(SNR_LEVELS))

    def snr_rows(mode):
        return "\n".join(
            f"| {n} | " + " | ".join(f"{v*100:.1f}" for v in S["noise"][mode][n]) + " |"
            for n in names)


    own = S.get("own_voice")
    if own:
        ocv = own["cv"]
        otop = max(ocv[n]["acc_mean"] for n in names)
        obest = [n for n in names if ocv[n]["acc_mean"] >= otop - 1e-9]
        rows_own = "\n".join(
            f"| {n} | {cv[n]['acc_mean']*100:.2f} | {ocv[n]['acc_mean']*100:.2f} ± "
            f"{ocv[n]['acc_std']*100:.2f} | {ocv[n]['f1_mean']*100:.2f} |" for n in names)
        oseg = own["segmentation"]
        own_txt = f"""## 6. Part B — My own voice

The same pipeline was run on recordings of **my own voice**, so the models are
tested on a new speaker, a new language (English *yes* / *no* instead of Hebrew
*ken* / *lo*) and a real phone microphone.

| Property | Value |
|---|---|
| Recordings | {own['n_recordings']} files × 8 words = {own['n_words']} words ({own['n_yes']} YES / {own['n_no']} NO) |
| Words | English "yes" and "no", read from a fixed random script |
| Recording device | mobile phone (Telegram voice note, Opus 48 kHz), converted to {own['sr']} Hz mono WAV |
| Environment | everyday surroundings, not a studio |
| File names | same format as the course dataset, e.g. `1_0_0_1_1_0_1_0.wav` |
| Segmentation | exactly 8 words found in **{oseg['ok']} of {oseg['total']}** recordings |

![My voice](figures/fig08_my_voice_signal.png)

![My voice yes vs no](figures/fig09_my_voice_yes_vs_no.png)

In my recordings the English **"yes"** ends with the fricative **/s/**, which is
visible as a weak burst of high-frequency energy (about 4–8 kHz) at the end of
most YES words in the spectrogram; **"no"** is a voiced word with energy mainly below 1 kHz. This is a
different cue from the Hebrew words in the course dataset, so the Part A models
cannot be reused directly — each model was trained and tested again on my voice
with the same grouped cross-validation ({ocv[names[0]]['n_splits']} folds, whole
recordings held out).

| Model | Course dataset acc. (%) | My voice acc. (%) | My voice Macro-F1 (%) |
|---|---:|---:|---:|
{rows_own}

![Dataset vs my voice](figures/fig10_dataset_vs_my_voice.png)

Best on my voice: {", ".join(f"**{n}**" for n in obest)}{" (tied)" if len(obest) > 1 else ""}
— {otop*100:.2f}%.

Accuracy on my voice is lower than on the course dataset, and the reasons are
clear: only {own['n_words']} words (about 6× fewer), a phone microphone with a
compressed voice-note codec, background sound from the room, and a weak /s/ that
carries much of the YES/NO difference but is only a small part of each word's
averaged MFCCs. The results also vary more from fold to fold (see the ± values),
so more recordings would give a more reliable ranking. Next steps: record more
words, add frame-level features (e.g. a CNN on the spectrogram) so that short
cues like /s/ count more, and apply the noise augmentation from Part A.

"""
    else:
        own_txt = ""
    seg = S["segmentation"]
    txt = f"""# Mini-Task 8 — Part 1: Spoken "Yes" / "No" Classification

**Course:** Digital Speech Processing — Dr. Khalooei
**Student:** Ghassan Alsafar — **ID:** 404131085

---

## 1. Goal

Classify isolated spoken words as **YES** or **NO**, compare several classifiers
(including **SVM** and a **Neural Network**), and report the results. Because
all models are close to perfect on clean audio, a **noise-robustness study** was
added to show which models really generalise, and the whole pipeline was also
run on **recordings of my own voice** (Part B, section 6).

## 2. Dataset — OpenSLR `yesno`

| Property | Value |
|---|---|
| Source | OpenSLR resource 1 (`waves_yesno.tar.gz`) |
| Recordings | {S['n_recordings']} `.wav` files, one speaker |
| Words per recording | 8 (Hebrew *ken* = YES, *lo* = NO) |
| Sample rate | {S['sr']} Hz, mono |
| Average duration | {S['avg_dur']:.2f} s per recording |
| Labels | in the file name, e.g. `0_1_1_0_...wav` → NO, YES, YES, NO, … |
| Class balance | {S['n_yes']} YES / {S['n_no']} NO words |

![Signal and spectrogram](figures/fig01_signal_and_spectrogram.png)

The waveform shows eight clear bursts of energy separated by silence; the
spectrogram shows that each word has its own formant pattern, which is what
the MFCC features capture.

## 3. Method

```
 .wav ─► energy-based word segmentation ─► 8 words ─► MFCC + ΔMFCC (52-dim) ─► classifier ─► YES / NO
```

### 3.1 Word segmentation (energy-based VAD)

1. Short-time energy in dB (25 ms frames, 10 ms hop), lightly smoothed.
2. Adaptive threshold between the noise floor (10th percentile) and the speech
   peak (99th percentile); several levels are tried and the one that yields
   exactly 8 words is kept.
3. Gaps shorter than 150 ms are merged, blobs shorter than 100 ms are dropped,
   30 ms of padding is added before each word and a 150 ms **hangover** after it
   (never past the middle of the pause), so weak word endings such as the /s/
   of English "yes" are not cut off.

**Result:** exactly 8 words were found in **{seg['ok']} of {seg['total']}**
recordings. {"Recordings with a different count were excluded so that every word keeps its correct label: " + ", ".join("`" + n + "`" for n in seg['failed']) + "." if seg['failed'] else "No recording had to be excluded."}
Total words used: **{S['n_words']}**.

![Energy segmentation](figures/fig02_energy_segmentation.png)

### 3.2 Features — MFCC implemented from scratch

Pre-emphasis (0.97) → 25 ms Hamming frames → 512-point FFT → 26 Mel filters →
log → DCT → first 13 coefficients. Each word is summarised by the **mean and
standard deviation of the 13 MFCCs and of their deltas → 52 features**.

![Yes vs No](figures/fig03_yes_vs_no.png)

![PCA](figures/fig04_feature_space_pca.png)

Even in two PCA dimensions the two classes form two clearly separated clusters,
so high accuracy is expected.

### 3.3 Classifiers

| Model | Settings |
|---|---|
| SVM | RBF kernel, C = 10, standardised features |
| Neural Network (MLP) | 2 hidden layers (64, 32), ReLU, Adam, L2 = 1e-3 |
| KNN | k = 5, standardised features |
| Random Forest | 200 trees |
| Logistic Regression | L2, standardised features |
| Naive Bayes | Gaussian |

### 3.4 Evaluation protocol

* **Grouped 5-fold cross-validation by recording** — all 8 words of a recording
  are always in the same fold, so the test set never contains words from a
  recording seen during training.
* A fixed **70 / 30 split by recording** ({S['n_train_rec']} train /
  {S['n_test_rec']} test recordings) is used for the noise study.
* **Noise study:** white Gaussian noise is added to the test words at
  SNR = 20, 15, 10, 5, 0, −5 dB (average of {NOISE_REPEATS} random noise draws).
  Two training conditions are compared: *clean only* and *multi-condition*
  (each training word is also added once with noise at a random SNR between
  0 and 20 dB).

## 4. Results

### 4.1 Clean speech — 5-fold grouped cross-validation

| Model | Accuracy (%) | Macro-F1 (%) | Training time (ms) |
|---|---:|---:|---:|
{rows_cv}

![Model comparison](figures/fig05_model_comparison.png)

Best on clean speech: {best_clean_txt}.

### 4.2 Noise robustness — test accuracy (%)

**Trained on clean speech**

| Model | {snr_hdr} |
|---|{snr_sep}|
{snr_rows('clean_train')}

**Trained with noise augmentation (multi-condition)**

| Model | {snr_hdr} |
|---|{snr_sep}|
{snr_rows('multi_train')}

![Noise robustness](figures/fig07_noise_robustness.png)

Confusion matrices of the noise-augmented models on the held-out recordings at 5 dB SNR:

![Confusion matrices](figures/fig06_confusion_matrices_5dB.png)

## 5. Discussion

1. **The clean task is easy.** Two words from one speaker in a quiet room are
   well separated by MFCCs (see the PCA plot), so most classifiers reach
   (almost) 100 %. Clean accuracy alone therefore cannot rank the models.
2. **Noise separates the models.** Averaged over all noise levels (20 → −5 dB)
   with clean training, **{best_rob}** is the most robust
   ({rob[best_rob]*100:.1f}% mean accuracy) and **{worst_rob}** degrades the most
   ({rob[worst_rob]*100:.1f}%). Noise fills the silent parts and the high
   frequencies of each word, which shifts the MFCC statistics away from what a
   clean-trained model has seen.
3. **Noise augmentation helps a lot.** Adding noisy copies of the training words
   raises the mean noisy-test accuracy by **{gain:+.1f} points** on average over
   all models; the most robust model in that setting is **{best_rob_m}**
   ({rob_m[best_rob_m]*100:.1f}% mean, {a0m[best_rob_m]*100:.1f}% at 0 dB).
4. **SVM vs Neural Network.** Both are at the top on clean speech; under noise,
   SVM reaches {a0m['SVM (RBF)']*100:.1f}% and the MLP {a0m['Neural Network (MLP)']*100:.1f}%
   at 0 dB (multi-condition training). With only {S['n_words']} training words the
   small MLP has little data to learn from; a larger, multi-speaker dataset would
   be needed for a neural network to show a clear advantage over the SVM.

**Limitations.** One speaker and only two words; white noise only; word
boundaries for the noise study come from the clean signal. Next steps: more
speakers (e.g. Google Speech Commands), real background noise (babble, music)
and spectrogram-based CNNs.

{own_txt}## 7. How to run

```bash
pip install -r requirements.txt
python yesno_classification.py
```

The script downloads the data (~4.5 MB), then writes every figure to
`figures/`, every number to `results/results.json`, and regenerates this README.
Run time: about one minute on a laptop.

## 8. Files

```
MiniTask08_YesNo/
├── yesno_classification.py   # the whole pipeline
├── requirements.txt
├── README.md                 # this report (auto-generated)
├── MiniTask08_YesNo.ipynb    # notebook that runs the pipeline and shows the results
├── my_voice/                 # Part B: my own 10 recordings (80 words)
├── figures/                  # all figures
└── results/results.json      # all numbers
```

*Environment: Python {S['env']['python']}, scikit-learn {S['env']['sklearn']},
NumPy {S['env']['numpy']}.*
"""
    (HERE / "README.md").write_text(txt, encoding="utf-8")
    log("  saved README.md")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR,
                    help="folder that holds (or will hold) the yesno .wav files")
    ap.add_argument("--own-voice", type=Path, default=OWN_DIR,
                    help="folder with my own recordings (same file-name format)")
    args = ap.parse_args()
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    log("[1/7] Dataset")
    wav_dir = ensure_dataset(args.data_dir)
    recs = load_dataset(wav_dir)
    sr = recs[0]["sr"]
    log(f"  {len(recs)} recordings, sample rate {sr} Hz")

    log("[2/7] Word segmentation")
    words, _, y, groups, failed, seg_cache = segment_recordings(recs)
    log(f"  exactly 8 words in {len(recs) - len(failed)}/{len(recs)} recordings -> {len(words)} words")

    log("[3/7] Features + exploration figures")
    X = np.array([word_features(w, sr) for w in words])
    ex = next(r for r in recs if r["name"] not in failed and 0 < sum(r["labels"]) < 8)
    plot_overview(ex, seg_cache[ex["name"]][0])
    plot_segmentation(ex, *seg_cache[ex["name"]])
    i_yes, i_no = int(np.where(y == 1)[0][0]), int(np.where(y == 0)[0][0])
    plot_class_examples(words[i_yes], words[i_no], sr, X, y)
    plot_pca(X, y)

    log("[4/7] Classifier comparison (grouped 5-fold CV)")
    names = list(build_models())
    cv = grouped_cv(X, y, groups, names)
    plot_model_comparison(cv)

    log("[5/7] Noise robustness")
    tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED).split(X, y, groups))
    X_tr_clean = X[tr_idx]
    X_tr_aug = np.vstack([X_tr_clean, np.array([
        word_features(add_noise(words[i], rng.uniform(0, 20), rng), sr) for i in tr_idx])])
    y_tr_aug = np.concatenate([y[tr_idx], y[tr_idx]])
    trained = {"clean_train": {}, "multi_train": {}}
    for n in names:
        trained["clean_train"][n] = build_models()[n].fit(X_tr_clean, y[tr_idx])
        trained["multi_train"][n] = build_models()[n].fit(X_tr_aug, y_tr_aug)
    noise = {"clean_train": {n: [] for n in names}, "multi_train": {n: [] for n in names}}
    cms = {}
    for snr in SNR_LEVELS:
        reps = 1 if snr is None else NOISE_REPEATS
        acc_sum = {mode: {n: 0.0 for n in names} for mode in noise}
        for rep in range(reps):
            r2 = np.random.default_rng(1000 + rep + (0 if snr is None else 7 * snr))
            Xt = np.array([word_features(add_noise(words[i], snr, r2), sr) for i in te_idx])
            for mode in noise:
                for n in names:
                    p = trained[mode][n].predict(Xt)
                    acc_sum[mode][n] += accuracy_score(y[te_idx], p)
                    if snr == 5 and rep == 0 and mode == "multi_train":
                        cms[n] = confusion_matrix(y[te_idx], p, labels=[0, 1]).tolist()
        for mode in noise:
            for n in names:
                noise[mode][n].append(acc_sum[mode][n] / reps)
        log(f"  SNR {'clean' if snr is None else str(snr) + ' dB':>6}: " +
            ", ".join(f"{n.split(' ')[0]} {noise['clean_train'][n][-1]*100:.0f}%" for n in names))
    plot_noise(noise, names)
    plot_confusions(cms, names)

    log("[6/7] Part B - my own voice")
    own = None
    own_dir = find_own_voice(args.own_voice)
    if own_dir is None:
        log("  no my_voice/ recordings found - Part B skipped")
    else:
        orecs = load_dataset(own_dir)
        ow, osr, oy, og, ofail, ocache = segment_recordings(orecs)
        log(f"  {len(orecs)} recordings at {orecs[0]['sr']} Hz, exactly 8 words in "
            f"{len(orecs) - len(ofail)}/{len(orecs)} -> {len(ow)} words")
        OX = np.array([word_features(w, r) for w, r in zip(ow, osr)])
        oex = next((r for r in orecs if r["name"] not in ofail), orecs[0])
        plot_overview(oex, ocache[oex["name"]][0], fname="fig08_my_voice_signal.png", who="My recording")
        iy, ino = int(np.where(oy == 1)[0][0]), int(np.where(oy == 0)[0][0])
        plot_class_examples(ow[iy], ow[ino], osr[iy], OX, oy, fname="fig09_my_voice_yes_vs_no.png",
                            who="My voice - ")
        cv_own = grouped_cv(OX, oy, og, names)
        plot_dataset_vs_own(cv, cv_own, names)
        own = {"n_recordings": len(orecs), "sr": orecs[0]["sr"], "n_words": len(ow),
               "n_yes": int((oy == 1).sum()), "n_no": int((oy == 0).sum()),
               "avg_dur": float(np.mean([len(r["x"]) / r["sr"] for r in orecs])),
               "segmentation": {"ok": len(orecs) - len(ofail), "total": len(orecs), "failed": ofail},
               "cv": cv_own}

    log("[7/7] Report")
    summary = {
        "n_recordings": len(recs), "sr": sr,
        "avg_dur": float(np.mean([len(r["x"]) / r["sr"] for r in recs])),
        "n_words": len(words), "n_yes": int((y == 1).sum()), "n_no": int((y == 0).sum()),
        "segmentation": {"ok": len(recs) - len(failed), "total": len(recs), "failed": failed},
        "n_train_rec": len(set(groups[tr_idx])), "n_test_rec": len(set(groups[te_idx])),
        "model_names": names, "cv": cv,
        "snr_levels": ["clean" if s is None else s for s in SNR_LEVELS],
        "noise": noise, "confusion_5dB": cms, "own_voice": own,
        "env": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                "numpy": np.__version__},
    }
    (RES_DIR / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(summary)
    log(f"\nDone in {time.time() - t0:.0f} s. Open README.md to see the report.")


if __name__ == "__main__":
    main()
