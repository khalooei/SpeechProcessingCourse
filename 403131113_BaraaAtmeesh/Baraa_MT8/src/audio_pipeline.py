from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, stft


def frame_rms(signal, frame_length=200, hop=80, sample_rate=8000):
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < frame_length:
        raise ValueError("Audio is shorter than the analysis frame")
    frames = np.lib.stride_tricks.sliding_window_view(signal, frame_length)[::hop]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    times = np.arange(len(rms)) * hop / sample_rate
    return rms, times


def segment_words(signal, sample_rate=8000, expected_words=8):
    if sample_rate != 8000:
        raise ValueError(f"Expected 8000 Hz, received {sample_rate}")
    rms, times = frame_rms(signal, sample_rate=sample_rate)
    log_energy = gaussian_filter1d(20 * np.log10(rms + 1e-8), sigma=1.6)
    floor = float(np.percentile(log_energy, 15))
    candidates, properties = find_peaks(
        log_energy, distance=32, prominence=5.0, height=floor + 7.0
    )
    reason = "accepted"
    if len(candidates) < expected_words:
        reason = f"Only {len(candidates)} energy peaks were detected"
        return None, {"reason": reason, "energy": log_energy, "times": times, "peaks": candidates}
    if len(candidates) > expected_words:
        levels = log_energy[candidates]
        strongest = np.argsort(levels)[::-1]
        margin_db = float(levels[strongest[expected_words-1]] - levels[strongest[expected_words]])
        if margin_db < 7.0:
            reason = f"Extra peaks cannot be separated reliably (gap {margin_db:.1f} dB)"
            return None, {"reason": reason, "energy": log_energy, "times": times, "peaks": candidates}
        candidates = np.sort(candidates[strongest[:expected_words]])
        reason = f"accepted after rejecting weaker peaks (margin {margin_db:.1f} dB)"
    peaks = np.sort(candidates)
    boundaries = [0]
    for a, b in zip(peaks[:-1], peaks[1:]):
        boundaries.append(int(a + np.argmin(log_energy[a:b+1])))
    boundaries.append(len(log_energy) - 1)
    spans = []
    for i, p in enumerate(peaks):
        lo, hi = boundaries[i], boundaries[i+1]
        local_floor = float(np.percentile(log_energy[lo:hi+1], 10))
        threshold = local_floor + 0.27 * (log_energy[p] - local_floor)
        region = log_energy[lo:hi+1] >= threshold
        j = p - lo
        left = right = j
        while left > 0 and region[left-1]:
            left -= 1
        while right < len(region)-1 and region[right+1]:
            right += 1
        start = max(0, int((lo + left)*80 - 240))
        end = min(len(signal), int((lo + right)*80 + 200 + 240))
        duration = (end - start) / sample_rate
        if not (0.12 <= duration <= 0.85):
            reason = f"Suspicious word {i+1} duration: {duration:.3f} s"
            return None, {"reason": reason, "energy": log_energy, "times": times, "peaks": peaks}
        spans.append((start, end))
    if any(spans[i][1] > spans[i+1][0] + int(.10*sample_rate) for i in range(len(spans)-1)):
        reason = "Adjacent word regions overlap too much"
        return None, {"reason": reason, "energy": log_energy, "times": times, "peaks": peaks}
    return spans, {"reason": reason, "energy": log_energy, "times": times, "peaks": peaks}


def spectral_features(signal, sample_rate=8000):
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < 256:
        signal = np.pad(signal, (0, 256 - len(signal)))
    freqs, _, transform = stft(signal, fs=sample_rate, window="hann",
                                nperseg=256, noverlap=176, nfft=256,
                                boundary=None, padded=False)
    power = np.abs(transform) ** 2 + 1e-12
    band_edges = np.geomspace(100, 4000, 17)
    bands = []
    for j in range(16):
        inside = (freqs >= band_edges[j]) & (
            freqs < band_edges[j+1] if j < 15 else freqs <= band_edges[j+1])
        bands.append(np.log(np.mean(power[inside], axis=0) + 1e-12))
    bands = np.asarray(bands)
    magnitude = np.sqrt(power)
    total_magnitude = np.sum(magnitude, axis=0) + 1e-12
    centroid = np.sum(freqs[:,None]*magnitude, axis=0) / total_magnitude / 4000
    cumulative = np.cumsum(power, axis=0)
    rolloff = freqs[np.argmax(cumulative >= 0.85*np.sum(power,axis=0),axis=0)] / 4000
    normalized = magnitude / total_magnitude
    flux = np.r_[0, np.linalg.norm(np.diff(normalized,axis=1),axis=0)]
    framed = np.lib.stride_tricks.sliding_window_view(signal,256)[::80]
    zcr = np.mean(np.diff(np.signbit(framed),axis=1)!=0,axis=1)
    n_frames = len(centroid)
    zcr = zcr[:n_frames] if len(zcr)>=n_frames else np.pad(zcr,(0,n_frames-len(zcr)),mode="edge")
    tracks = np.vstack([bands,centroid,rolloff,flux,zcr])
    features = np.r_[np.mean(tracks,axis=1),np.std(tracks,axis=1)]
    if features.shape != (40,) or not np.isfinite(features).all():
        raise ValueError("Feature extraction produced an invalid vector")
    return features
