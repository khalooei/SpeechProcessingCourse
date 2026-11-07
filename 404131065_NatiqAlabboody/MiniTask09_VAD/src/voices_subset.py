"""Loading, grouping and reference labels for the VOiCES 90-degree subset (Lab41/VOiCES-subset).

The subset has no clean source files and no frame labels, so the speech reference is built
from the recording of the same sentence, in the same room, with no distractor ("none").
That recording is labelled with an energy rule, then shifted onto every other recording of
the group by cross-correlating the speech envelopes.
"""
import os
import re
from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, uniform_filter1d
from scipy.signal import butter, fftconvolve, sosfiltfilt

import vad_utils as vu

NAME = re.compile(r"VOiCES-(rm\d)-(\w+?)-sp(\d+)-ch(\d+)-sg(\d+)-mc(\d+)-(\w+)-(\w+)-dg(\d+)")
MIC_ORDER = {"clo": 0, "mid": 1, "far": 2, "beh": 3}
_BAND = butter(4, [300, 3400], btype="band", fs=vu.SR, output="sos")


def index(folder):
    rows = []
    for f in sorted(os.listdir(folder)):
        m = NAME.search(f)
        if f.endswith(".wav") and not f.startswith("._") and m:
            room, dist, sp, ch, sg, mc, mtype, mloc, _ = m.groups()
            rows.append({"path": os.path.join(folder, f), "file": f, "room": room, "distractor": dist,
                         "speaker": sp, "utt": f"{sp}-{ch}-{sg}", "mic": mc, "mic_loc": mloc})
    return rows


def band_env(x):
    """Log envelope of the 300-3400 Hz band, one value per 10 ms frame."""
    y = sosfiltfilt(_BAND, x)
    return 10 * np.log10(np.mean(vu.frames(y) ** 2, axis=1) + 1e-12)


def reference_mask(x):
    """Speech frames of a no-distractor recording: smoothed band energy above floor + 30% of range."""
    e = uniform_filter1d(band_env(x), 5)
    lo, hi = np.percentile(e, 5), np.percentile(e, 98)
    m = e > lo + 0.3 * (hi - lo)
    return binary_opening(binary_closing(m, np.ones(9)), np.ones(5))


def lag_frames(ref, x, max_s=2.0):
    """Frames by which `ref` must be delayed to line up with `x` (positive: x starts later)."""
    a = band_env(x); b = band_env(ref)
    a = np.clip(a - np.percentile(a, 30), 0, None); b = np.clip(b - np.percentile(b, 30), 0, None)
    a -= a.mean(); b -= b.mean()
    c = fftconvolve(a, b[::-1]); zero = len(b) - 1; m = int(max_s * 100)
    lo = max(zero - m, 0)
    w = c[lo:zero + m + 1]
    k = int(np.argmax(w))
    peak = w[k] / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return k + lo - zero, float(peak)


def shift(mask, lag, n):
    out = np.zeros(n, bool)
    a, b = max(lag, 0), max(-lag, 0)
    m = min(n - a, len(mask) - b)
    if m > 0:
        out[a:a + m] = mask[b:b + m]
    return out


def groups(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["utt"], r["room"])].append(r)
    return g


def labelled_set(rows, load=vu.load_audio):
    """For every group with a 'none' recording: (row, audio, frame labels, lag, peak) for all members."""
    out = []
    for key, members in groups(rows).items():
        refs = sorted([r for r in members if r["distractor"] == "none"], key=lambda r: MIC_ORDER[r["mic_loc"]])
        if not refs:
            continue
        ref = refs[0]
        xr = load(ref["path"])
        mr = reference_mask(xr)
        for r in members:
            x = xr if r is ref else load(r["path"])
            n = 1 + (max(len(x), vu.FRAME) - vu.FRAME) // vu.HOP
            lag, peak = (0, 1.0) if r is ref else lag_frames(xr, x)
            out.append({"row": r, "x": x, "y": shift(mr, lag, n), "lag": lag, "peak": peak, "ref": ref["file"]})
    return out
