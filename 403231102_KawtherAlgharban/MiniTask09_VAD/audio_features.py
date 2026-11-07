"""
Shared helpers for Mini-Task 9 (VOiCES): reading audio, file-name parsing,
framing, MFCC and simple spectral features - all implemented with NumPy.
Digital Speech Processing - Dr. Mohammad Khalooei - AUT
Student: Kawther Algharban (403231102)
"""

import re
import wave
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "data" / "VOiCES-subset" / "audio"
FIG_DIR = BASE_DIR / "figures"
RES_DIR = BASE_DIR / "results"
FEAT_DIR = BASE_DIR / "features"

SR = 16000
FRAME = 400        # 25 ms
HOP = 160          # 10 ms
N_FFT = 512

CLASSES = ["none", "babb", "musi", "tele"]
CLASS_NAMES = {"none": "no distractor", "babb": "babble",
               "musi": "music", "tele": "television"}

NAME_RE = re.compile(
    r"(?P<room>rm\d)-(?P<distractor>none|babb|musi|tele)"
    r"-sp(?P<speaker>\d+)-ch(?P<chapter>\d+)-se?g(?P<segment>\d+)"
    r"-mc(?P<mic>\d+)-(?P<mic_type>[a-z]+)-(?P<mic_loc>[a-z0-9]+)"
)


def list_files():
    files = []
    for p in sorted(AUDIO_DIR.glob("*.wav")):
        m = NAME_RE.search(p.name)
        if m:
            info = m.groupdict()
            info["path"] = p
            info["sentence"] = f"{info['speaker']}-{info['chapter']}-{info['segment']}"
            files.append(info)
    if not files:
        raise SystemExit("No audio found. Run step1_explore_voices.py first.")
    return files


def load_wav(path):
    with wave.open(str(path), "rb") as f:
        sr, ch, width = f.getframerate(), f.getnchannels(), f.getsampwidth()
        raw = f.readframes(f.getnframes())
    x = np.frombuffer(raw, dtype={2: np.int16, 4: np.int32}[width]).astype(np.float32)
    x /= float(2 ** (8 * width - 1))
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def frame_signal(x, frame=FRAME, hop=HOP):
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


# ---------------- Mel filterbank + DCT ----------------
def _hz2mel(f):
    return 2595 * np.log10(1 + f / 700)


def _mel2hz(m):
    return 700 * (10 ** (m / 2595) - 1)


def mel_filterbank(n_filters=40, n_fft=N_FFT, sr=SR):
    mels = np.linspace(_hz2mel(0), _hz2mel(sr / 2), n_filters + 2)
    bins = np.floor((n_fft + 1) * _mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(r - c, 1)
    return fb


def dct_matrix(n_out, n_in):
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    d = np.cos(np.pi * k * (2 * n + 1) / (2 * n_in)) * np.sqrt(2 / n_in)
    d[0] /= np.sqrt(2)
    return d


FBANK = mel_filterbank()
DCT = dct_matrix(13, FBANK.shape[0])
WINDOW = np.hamming(FRAME)
FREQS = np.fft.rfftfreq(N_FFT, 1 / SR)


def frame_features(x):
    """Per-frame features, shape (n_frames, 19):
       13 MFCC, log energy, spectral centroid, bandwidth, roll-off (85%),
       spectral flatness, zero-crossing rate."""
    frames_raw = frame_signal(x)
    xe = np.append(x[0], x[1:] - 0.97 * x[:-1])                 # pre-emphasis
    frames = frame_signal(xe) * WINDOW
    power = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2 / N_FFT + 1e-12
    mfcc = np.log(power @ FBANK.T + 1e-10) @ DCT.T

    log_e = 10 * np.log10(np.mean(frames_raw ** 2, axis=1) + 1e-10)
    p_sum = power.sum(axis=1)
    centroid = (power * FREQS).sum(axis=1) / p_sum
    bandwidth = np.sqrt((power * (FREQS - centroid[:, None]) ** 2).sum(axis=1) / p_sum)
    cum = np.cumsum(power, axis=1)
    rolloff = FREQS[np.argmax(cum >= 0.85 * cum[:, -1:], axis=1)]
    flatness = np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)
    zcr = np.mean(np.abs(np.diff(np.sign(frames_raw), axis=1)) > 0, axis=1)

    extra = np.stack([log_e, centroid / 1000, bandwidth / 1000, rolloff / 1000,
                      flatness, zcr], axis=1)
    return np.hstack([mfcc, extra]).astype(np.float32)


FEATURE_NAMES = [f"mfcc{i}" for i in range(13)] + [
    "log_energy", "centroid_kHz", "bandwidth_kHz", "rolloff_kHz", "flatness", "zcr"]
