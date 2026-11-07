"""Run the energy VAD on one wav file and save a plot and a list of speech segments.

    python vad_file.py --input recording.wav --out result

This shows what the rule detects. It is not an accuracy result: that needs reference labels.
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import vad_utils as vu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="vad_result")
    ap.add_argument("--alpha", type=float, default=0.4, help="threshold position between floor and peak")
    a = ap.parse_args()
    x = vu.load_audio(a.input)
    d = vu.energy_vad(x, a.alpha).astype(bool)
    edges = np.diff(d.astype(int), prepend=0, append=0)
    seg = [(s * vu.HOP / vu.SR, e * vu.HOP / vu.SR) for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])]
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s"])
        w.writerows([[round(s, 2), round(e, 2)] for s, e in seg])
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.plot(np.arange(len(x)) / vu.SR, x, color="#14305c", lw=0.3)
    for s, e in seg:
        ax.axvspan(s, e, color="#c8981e", alpha=0.3, lw=0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.savefig(a.out + ".png", dpi=160, bbox_inches="tight")
    print(f"{len(seg)} speech segments, {sum(e - s for s, e in seg):.2f} s of {len(x) / vu.SR:.2f} s")


if __name__ == "__main__":
    main()
