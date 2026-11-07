"""Front-end, VAD rules and small helpers for the events / VAD mini task."""
from math import gcd

import numpy as np
from scipy.fft import dct
from scipy.io import wavfile
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import fftconvolve, resample_poly

SR = 8000           # VOiCES is 16 kHz; run_on_voices.py resamples it to this rate
FRAME = int(0.025 * SR)
HOP = int(0.010 * SR)
NFFT = 256


def load_audio(path, sr=SR):
    fs, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x /= 32768.0
    if fs != sr:
        g = gcd(fs, sr)
        x = resample_poly(x, sr // g, fs // g)
    return x


def frames(x):
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)))
    n = 1 + (len(x) - FRAME) // HOP
    return x[np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]]


def _mel_bank(n_mels=26):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    b = np.floor((NFFT + 1) * mel2hz(np.linspace(0, hz2mel(SR / 2), n_mels + 2)) / SR).astype(int)
    fb = np.zeros((n_mels, NFFT // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = b[m - 1], b[m], b[m + 1]
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


_FB, _WIN = _mel_bank(), np.hamming(FRAME)


def power_spec(x):
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    return np.abs(np.fft.rfft(frames(x) * _WIN, NFFT)) ** 2 / NFFT


def mfcc(x, n=13):
    return dct(np.log(power_spec(x) @ _FB.T + 1e-10), type=2, axis=1, norm="ortho")[:, :n]


def delta(c, w=2):
    p = np.pad(c, ((w, w), (0, 0)), mode="edge")
    return sum(i * (p[w + i:len(p) - w + i] - p[w - i:len(p) - w - i])
               for i in range(1, w + 1)) / (2 * sum(i * i for i in range(1, w + 1)))


def clip_features(x):
    """Mean and std of MFCC + delta over a clip; c0 mean removed for level invariance."""
    c = mfcc(x)
    c[:, 0] -= c[:, 0].mean()
    a = np.hstack([c, delta(c)])
    return np.r_[a.mean(axis=0), a.std(axis=0)]


def frame_features(x):
    """Per-frame MFCC + deltas, mean-normalised over the stream, plus a 310 ms local mean."""
    c = mfcc(x)
    c -= c.mean(axis=0)
    a = np.hstack([c, delta(c), delta(delta(c))])
    return np.hstack([a, uniform_filter1d(a, size=31, axis=0)])


def smooth(decisions, ms=200):
    """Median filter the binary frame decisions."""
    return median_filter(decisions.astype(int), size=int(ms / 10) | 1)


def energy_vad(x, alpha=0.35):
    """Classic rule: log energy above floor + alpha * (peak - floor)."""
    e = 10 * np.log10(np.mean(frames(x) ** 2, axis=1) + 1e-12)
    e = uniform_filter1d(e, 5)
    lo, hi = np.percentile(e, 10), np.percentile(e, 95)
    return smooth(e > lo + alpha * (hi - lo))


def entropy_vad(x, alpha=0.5):
    """Spectral entropy rule: voiced speech has a peaky, low-entropy spectrum."""
    p = power_spec(x)[:, 8:]                  # above 250 Hz, skipping hum
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    h = uniform_filter1d(-(p * np.log(p + 1e-12)).sum(axis=1), 5)
    lo, hi = np.percentile(h, 5), np.percentile(h, 90)
    return smooth(h < lo + alpha * (hi - lo))


def reverb(x, rng):
    """Very simple room model: exponentially decaying noise tail, RT60 of 0.3-0.6 s."""
    rt60 = rng.uniform(0.3, 0.6)
    t = np.arange(int(rt60 * SR)) / SR
    rir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60) * 0.15
    rir[0] = 1.0
    y = fftconvolve(x, rir)[:len(x)]
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-12) * np.sqrt(np.mean(x ** 2))


def scale_to_snr(speech, noise, snr_db):
    """Gain for `noise` so that active speech power over noise power equals snr_db."""
    active = speech[np.abs(speech) > 0.01 * np.abs(speech).max()]
    return np.sqrt(np.mean(active ** 2) / (10 ** (snr_db / 10) * (np.mean(noise ** 2) + 1e-12)))


def frame_scores(y, p):
    y, p = y.astype(bool), p.astype(bool)
    tp, fp, fn, tn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum(), (~y & ~p).sum()
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def from_counts(c):
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    return {"accuracy": (tp + tn) / (tp + fp + fn + tn),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "miss_rate": fn / max(tp + fn, 1),
            "false_alarm_rate": fp / max(fp + tn, 1)}
