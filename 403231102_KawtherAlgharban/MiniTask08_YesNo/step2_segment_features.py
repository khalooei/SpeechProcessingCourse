"""
Mini-Task 8 - Step 2: Word segmentation (simple VAD) + MFCC features
Digital Speech Processing - Dr. Mohammad Khalooei - AUT
Student: Kawther Algharban (403231102)

Each YesNo file contains 8 words, but the label is given for the whole file.
To classify single words we must first cut every file into its 8 words.

Pipeline (same order as the Signal Lab page of the course):
  signal -> framing (25 ms, hop 10 ms) -> short-time energy -> peak picking (VAD)
         -> 8 word segments -> pre-emphasis -> Hamming window -> FFT
         -> Mel filterbank (26 filters) -> log -> DCT -> 13 MFCC (+ deltas)
         -> fixed-length vector per word (mean + std over time)

Output:
  features/yesno_features.npz   (X, y, file_id)  -> used by step 3
  figures/step2_segmentation.png
  figures/step2_segmentation_check.png   (6 more files, to verify visually)
  figures/step2_mfcc_yes_vs_no.png
"""

import wave
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
WAV_DIR = BASE_DIR / "data" / "waves_yesno"
FIG_DIR = BASE_DIR / "figures"
FEAT_DIR = BASE_DIR / "features"

SR = 8000
FRAME_LEN = 200          # 25 ms at 8 kHz
HOP = 80                 # 10 ms at 8 kHz
N_WORDS = 8


# ==================================================================
# 1. Reading audio
# ==================================================================
def load_wav(path):
    with wave.open(str(path), "rb") as f:
        sr = f.getframerate()
        frames = f.readframes(f.getnframes())
    x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


def labels_from_name(path):
    return [int(v) for v in path.stem.split("_")]


def frame_signal(x, frame_len=FRAME_LEN, hop=HOP):
    """Split a signal into overlapping frames -> shape (n_frames, frame_len)."""
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n = 1 + (len(x) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


# ==================================================================
# 2. Simple energy-based VAD -> exactly 8 word segments
# ==================================================================
def short_time_log_energy(x):
    frames = frame_signal(x)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-10)
    log_e = 20 * np.log10(rms)
    # smooth with a 5-frame moving average to remove small spikes
    return np.convolve(log_e, np.ones(5) / 5, mode="same")


def segment_words(x, n_words=N_WORDS, pad_frames=3, half_win=20, rel=0.30):
    """Return a list of (start_sample, end_sample) for each word.

    Method (energy peak picking):
      1. Every word is one "hill" in the energy curve, so we look for
         local maxima that are the highest point within +-200 ms.
      2. We keep the 8 highest hills (short clicks and breaths are much
         weaker than real words, so they are ignored).
      3. Between two neighbouring words the boundary is the lowest energy
         point (the silence between them).
      4. Inside its region each word is trimmed to the frames whose energy
         is above  floor + 30% * (word peak - floor).
    """
    e = short_time_log_energy(x)
    n = len(e)
    floor = np.percentile(e, 10)

    # ignore the first/last 100 ms (edge effects of the smoothing filter)
    search = e.copy()
    search[:10] = floor
    search[-10:] = floor

    peaks = [i for i in range(n)
             if search[i] == search[max(0, i - half_win):i + half_win + 1].max()
             and search[i] > floor + 6]                      # at least 6 dB above noise
    peaks = sorted(sorted(peaks, key=lambda i: search[i])[-n_words:])

    # boundaries = energy minimum between consecutive peaks
    cuts = [0] + [p + int(np.argmin(e[p:q])) for p, q in zip(peaks[:-1], peaks[1:])] + [n]

    segs = []
    for k, p in enumerate(peaks):
        lo, hi = cuts[k], cuts[k + 1]
        thr = floor + rel * (e[p] - floor)
        a = p
        while a > lo and e[a - 1] > thr:
            a -= 1
        b = p
        while b < hi - 1 and e[b + 1] > thr:
            b += 1
        a = max(lo, a - pad_frames)
        b = min(hi, b + pad_frames)
        segs.append((a * HOP, min(len(x), b * HOP + FRAME_LEN)))
    return segs, e


# ==================================================================
# 3. MFCC (implemented step by step with NumPy)
# ==================================================================
def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)


def mel_filterbank(n_filters=26, n_fft=256, sr=SR, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / sr).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb


def dct_matrix(n_out, n_in):
    """DCT-II matrix (orthonormal)."""
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    d = np.cos(np.pi * k * (2 * n + 1) / (2 * n_in)) * np.sqrt(2 / n_in)
    d[0] /= np.sqrt(2)
    return d


N_FFT = 256
FBANK = mel_filterbank(n_fft=N_FFT)
DCT = dct_matrix(13, FBANK.shape[0])
WINDOW = np.hamming(FRAME_LEN)


def mfcc(x):
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])          # pre-emphasis
    frames = frame_signal(x) * WINDOW                     # framing + window
    power = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2 / N_FFT   # power spectrum
    mel_energy = np.log(power @ FBANK.T + 1e-10)          # Mel + log
    return mel_energy @ DCT.T                             # DCT -> (frames, 13)


def deltas(c, width=2):
    padded = np.pad(c, ((width, width), (0, 0)), mode="edge")
    num = sum(n * (padded[width + n:len(c) + width + n] - padded[width - n:len(c) + width - n])
              for n in range(1, width + 1))
    return num / (2 * sum(n * n for n in range(1, width + 1)))


def word_features(seg):
    """Fixed-length vector: mean + std of 13 MFCC and 13 deltas = 52 numbers."""
    c = mfcc(seg)
    d = deltas(c)
    feats = np.hstack([c, d])
    return np.concatenate([feats.mean(axis=0), feats.std(axis=0)])


# ==================================================================
# 4. Main
# ==================================================================
def main():
    files = sorted(WAV_DIR.glob("*.wav"))
    if not files:
        raise SystemExit("No data found. Run step1_explore.py first.")

    X, y, file_id = [], [], []
    ok = 0
    bad_files = []

    for i, p in enumerate(files):
        x, sr = load_wav(p)
        labels = labels_from_name(p)
        segs, _ = segment_words(x)
        if len(segs) != N_WORDS:
            bad_files.append((p.name, len(segs)))
            continue
        ok += 1
        for (a, b), lab in zip(segs, labels):
            X.append(word_features(x[a:b]))
            y.append(lab)
            file_id.append(i)

    X, y, file_id = np.array(X), np.array(y), np.array(file_id)

    print("========== Segmentation ==========")
    print(f"Files segmented into exactly 8 words : {ok} / {len(files)}")
    for name, n in bad_files:
        print(f"   skipped {name}: found {n} segments")
    print("\n========== Features ==========")
    print("Feature matrix X :", X.shape, "(words x features)")
    print("Labels y         :", y.shape, f"(yes = {y.sum()}, no = {len(y) - y.sum()})")

    FEAT_DIR.mkdir(exist_ok=True)
    np.savez(FEAT_DIR / "yesno_features.npz", X=X, y=y, file_id=file_id)
    print("Saved to:", FEAT_DIR / "yesno_features.npz")

    # ---------------- Figure 1: segmentation of one file ----------------
    FIG_DIR.mkdir(exist_ok=True)
    p = files[0]
    x, sr = load_wav(p)
    segs, e = segment_words(x)
    labels = labels_from_name(p)
    t = np.arange(len(x)) / sr
    te = (np.arange(len(e)) * HOP + FRAME_LEN / 2) / sr

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(t, x, lw=0.5, color="gray")
    for (a, b), lab in zip(segs, labels):
        color = "tab:green" if lab == 1 else "tab:red"
        ax[0].axvspan(a / sr, b / sr, color=color, alpha=0.25)
        ax[0].text((a + b) / 2 / sr, 0.9 * np.abs(x).max(),
                   "YES" if lab else "NO", ha="center", color=color, weight="bold")
    ax[0].set_title(f"Automatic word segmentation: {p.name}")
    ax[0].set_ylabel("Amplitude")
    ax[1].plot(te, e)
    ax[1].set_title("Short-time log energy (dB) used by the VAD")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("dB")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step2_segmentation.png", dpi=150)

    # ---------------- Figure 1b: check segmentation on 6 other files ----------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 8))
    for axk, p2 in zip(axes.ravel(), files[1::10][:6]):
        x2, _ = load_wav(p2)
        segs2, _ = segment_words(x2)
        axk.plot(np.arange(len(x2)) / sr, x2, lw=0.4, color="gray")
        for (a, b), lab in zip(segs2, labels_from_name(p2)):
            axk.axvspan(a / sr, b / sr, color="tab:green" if lab else "tab:red", alpha=0.25)
        axk.set_title(p2.name, fontsize=9)
        axk.set_yticks([])
    fig.suptitle("Segmentation check (green = YES, red = NO)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step2_segmentation_check.png", dpi=150)

    # ---------------- Figure 2: MFCC of a YES word vs a NO word ----------------
    yes_idx = labels.index(1)
    no_idx = labels.index(0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k, (idx, name) in enumerate([(no_idx, "NO  ('lo')"), (yes_idx, "YES  ('ken')")]):
        a, b = segs[idx]
        im = ax[k].imshow(mfcc(x[a:b]).T, aspect="auto", origin="lower", cmap="viridis")
        ax[k].set_title(f"MFCC of a {name} word")
        ax[k].set_xlabel("Frame (10 ms)")
        ax[k].set_ylabel("MFCC coefficient")
        plt.colorbar(im, ax=ax[k])
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step2_mfcc_yes_vs_no.png", dpi=150)
    print("Figures saved to:", FIG_DIR)
    plt.show()


if __name__ == "__main__":
    main()
