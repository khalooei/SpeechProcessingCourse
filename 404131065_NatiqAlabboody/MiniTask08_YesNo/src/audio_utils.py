"""Shared helpers for the Yes/No mini task: loading, word segmentation, MFCC.

MFCCs are written out with numpy/scipy on purpose. It keeps the dependency list
short and it means every step of the front-end is visible in this file.
"""
import glob
import os
import tarfile
import urllib.request

import numpy as np
from scipy.fft import dct
from scipy.io import wavfile
from scipy.ndimage import binary_closing, binary_opening, uniform_filter1d

SAMPLE_RATE = 8000
FRAME_LEN = int(0.025 * SAMPLE_RATE)   # 25 ms -> 200 samples
HOP_LEN = int(0.010 * SAMPLE_RATE)     # 10 ms -> 80 samples
NFFT = 256
N_MELS = 26
N_MFCC = 13
PRE_EMPHASIS = 0.97

URLS = [
    "https://khalooei.ir/dataset/waves_yesno.tar.gz",        # the course tutorial's copy
    "https://www.openslr.org/resources/1/waves_yesno.tar.gz",
]


def ensure_dataset(data_dir):
    """Return the waves_yesno folder, downloading it if it is not there yet."""
    folder = os.path.join(data_dir, "waves_yesno")
    if len(glob.glob(os.path.join(folder, "*.wav"))) == 60:
        return folder
    os.makedirs(data_dir, exist_ok=True)
    archive = os.path.join(data_dir, "waves_yesno.tar.gz")
    for url in URLS:
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, archive)
            with tarfile.open(archive) as tar:
                try:
                    tar.extractall(data_dir, filter="data")
                except TypeError:  # Python versions without the filter argument
                    tar.extractall(data_dir)
            os.remove(archive)
            return folder
        except Exception as err:  # try the next mirror
            print(f"  failed: {err}")
    raise RuntimeError("Could not download waves_yesno. Put the 60 .wav files in " + folder)


def load_wav(path):
    """Read a wav file and return (float signal in [-1, 1], sample rate)."""
    sr, x = wavfile.read(path)
    return x.astype(np.float64) / 32768.0, sr


def labels_from_name(path):
    """'0_1_1_0_0_1_1_0.wav' -> [0, 1, 1, 0, 0, 1, 1, 0] (1 = yes, 0 = no)."""
    return [int(c) for c in os.path.basename(path)[:-4].split("_")]


def frame_signal(x, frame_len=FRAME_LEN, hop=HOP_LEN):
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n = 1 + (len(x) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def short_time_log_energy(x):
    frames = frame_signal(x)
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)


def segment_words(x, expected=8):
    """Find the `expected` words of a recording with an adaptive energy threshold.

    Recordings differ in level, so the threshold is searched rather than fixed.
    Every threshold between floor and peak is tried, and the middle of the range
    that gives exactly `expected` regions is kept: a threshold near the edge of
    that range would clip weak word endings. Returns a list of (start, end)
    sample indices, or None if no threshold works.
    """
    env = uniform_filter1d(short_time_log_energy(x), size=5)
    floor, peak = np.percentile(env, 10), np.percentile(env, 99)

    def regions(frac):
        mask = env > floor + frac * (peak - floor)
        # closing bridges gaps under 150 ms, opening drops clicks under 100 ms
        mask = binary_opening(binary_closing(mask, np.ones(15)), np.ones(10))
        edges = np.diff(mask.astype(int), prepend=0, append=0)
        return np.where(edges == 1)[0], np.where(edges == -1)[0]

    good = [f for f in np.linspace(0.9, 0.05, 171) if len(regions(f)[0]) == expected]
    if not good:
        return None
    starts, ends = regions(float(np.median(good)))
    pad = 5  # 50 ms either side
    return [(max(0, s - pad) * HOP_LEN, min(len(x), (e + pad) * HOP_LEN + FRAME_LEN))
            for s, e in zip(starts, ends)]


def mel_filterbank(sr=SAMPLE_RATE, nfft=NFFT, n_mels=N_MELS):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2))
    bins = np.floor((nfft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, nfft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


_FB = mel_filterbank()
_WIN = np.hamming(FRAME_LEN)


def mfcc(x):
    """Return MFCCs with shape (frames, N_MFCC)."""
    x = np.append(x[0], x[1:] - PRE_EMPHASIS * x[:-1])
    power = np.abs(np.fft.rfft(frame_signal(x) * _WIN, NFFT)) ** 2 / NFFT
    log_mel = np.log(power @ _FB.T + 1e-10)
    return dct(log_mel, type=2, axis=1, norm="ortho")[:, :N_MFCC]


def log_mel(x):
    """Log energies of the 26 mel filters, shape (frames, N_MELS). MFCCs add a DCT to this."""
    x = np.append(x[0], x[1:] - PRE_EMPHASIS * x[:-1])
    power = np.abs(np.fft.rfft(frame_signal(x) * _WIN, NFFT)) ** 2 / NFFT
    return np.log(power @ _FB.T + 1e-10)


def delta(c, width=2):
    """Regression deltas over +/- `width` frames."""
    pad = np.pad(c, ((width, width), (0, 0)), mode="edge")
    denom = 2 * sum(i * i for i in range(1, width + 1))
    return sum(i * (pad[width + i:len(pad) - width + i] - pad[width - i:len(pad) - width - i])
               for i in range(1, width + 1)) / denom


def word_features(x):
    """Fixed-size vector: mean and std of MFCC + delta + delta-delta (78 values)."""
    c = mfcc(x)
    c = c - c.mean(axis=0, keepdims=True) * np.r_[1, np.zeros(N_MFCC - 1)]  # c0 mean removal: level invariance
    d, dd = delta(c), delta(delta(c))
    all_c = np.hstack([c, d, dd])
    return np.r_[all_c.mean(axis=0), all_c.std(axis=0)]


def fbank_features(x):
    """Same recipe as word_features, on log filter bank energies (156 values)."""
    f = log_mel(x)
    f = f - f.mean()  # one level per word, the counterpart of removing the c0 mean
    d, dd = delta(f), delta(delta(f))
    all_f = np.hstack([f, d, dd])
    return np.r_[all_f.mean(axis=0), all_f.std(axis=0)]


def simple_features(x):
    """Baseline of 3 time-domain numbers: duration (s), mean ZCR, RMS in dB."""
    frames = frame_signal(x)
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
    return np.array([len(x) / SAMPLE_RATE, zcr.mean(),
                     10 * np.log10(np.mean(x ** 2) + 1e-12)])


def add_noise(x, snr_db, rng):
    """Add white Gaussian noise at a chosen SNR, measured over this segment."""
    noise = rng.standard_normal(len(x))
    gain = np.sqrt(np.mean(x ** 2) / (10 ** (snr_db / 10) * np.mean(noise ** 2)))
    return x + gain * noise
