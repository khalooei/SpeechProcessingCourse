"""Figures for the Yes/No report. Run after run_experiments.py."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import audio_utils as au

DATA = sys.argv[1] if len(sys.argv) > 1 else "../data"
RES = sys.argv[2] if len(sys.argv) > 2 else "../results"
OUT = os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b2b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": False, "savefig.dpi": 220, "savefig.bbox": "tight", "pdf.fonttype": 42})
NAVY = "#14305c"
GOLD = "#c8981e"
SEQ = LinearSegmentedColormap.from_list("navy", ["#ffffff", "#9fb6d8", "#3f6aa8", NAVY, "#071a36"])
STYLE = {"SVM (RBF)": ("#2a78d6", "o"), "Neural network (MLP)": ("#eb6834", "s"),
         "Logistic regression": ("#1baf7a", "^"), "Random forest": ("#eda100", "D"),
         "k-NN": ("#e87ba4", "v"), "Gaussian naive Bayes": ("#008300", "P")}
res = json.load(open(os.path.join(RES, "results.json")))


def spec_db(x):
    p = np.abs(np.fft.rfft(au.frame_signal(x) * np.hamming(au.FRAME_LEN), au.NFFT)) ** 2
    return 10 * np.log10(p + 1e-8)


# Figure 1: one recording
name = "0_1_1_0_0_1_1_0.wav"
x, sr = au.load_wav(os.path.join(DATA, "waves_yesno", name))
seg = au.segment_words(x)
lab = au.labels_from_name(name)
fig, ax = plt.subplots(2, 1, figsize=(6.6, 3.5), sharex=True, height_ratios=[1, 1.5])
t = np.arange(len(x)) / sr
ax[0].plot(t, x, color=NAVY, lw=0.4)
top = np.abs(x).max() * 1.15
for (a, b), l in zip(seg, lab):
    ax[0].axvspan(a / sr, b / sr, color=GOLD, alpha=0.22, lw=0)
    ax[0].text((a + b) / 2 / sr, top, "yes" if l else "no", ha="center", va="bottom", fontsize=9)
ax[0].set_ylabel("Amplitude")
S = spec_db(x)
ax[1].imshow(S.T, origin="lower", aspect="auto", cmap=SEQ, vmin=S.max() - 70, vmax=S.max(),
             extent=[0, S.shape[0] * au.HOP_LEN / sr, 0, sr / 2 / 1000])
ax[1].set_ylabel("Frequency (kHz)")
ax[1].set_xlabel("Time (s)")
fig.savefig(os.path.join(OUT, "fig1_recording.pdf"))
plt.close(fig)

# Figure 2: one yes against one no
fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.3), sharey=True)
for i, want in enumerate([1, 0]):
    a, b = seg[lab.index(want)]
    S = spec_db(x[a:b])
    ax[i].imshow(S.T, origin="lower", aspect="auto", cmap=SEQ, vmin=S.max() - 70, vmax=S.max(),
                 extent=[0, S.shape[0] * 10, 0, 4])
    ax[i].set_title("yes (“ken”)" if want else "no (“lo”)", fontsize=10)
    ax[i].set_xlabel("Time (ms)")
ax[0].set_ylabel("Frequency (kHz)")
fig.savefig(os.path.join(OUT, "fig2_yes_no.pdf"))
plt.close(fig)

# Figure 3: PCA
z = np.load(os.path.join(RES, "pca.npz"))
fig, ax = plt.subplots(figsize=(4.2, 2.8))
for v, c, m, n in [(1, "#2a78d6", "o", "yes"), (0, "#eb6834", "s", "no")]:
    ax.scatter(*z["Z"][z["y"] == v].T, s=15, c=c, marker=m, edgecolors="white", linewidths=0.4, label=n)
ax.set_xlabel(f"PC1 ({res['pca_variance'][0]*100:.0f}% of variance)")
ax.set_ylabel(f"PC2 ({res['pca_variance'][1]*100:.0f}%)")
ax.legend(frameon=False)
fig.savefig(os.path.join(OUT, "fig3_pca.pdf"))
plt.close(fig)

# Figure 4: noise, as two panels on one shared scale
snrs = ["20", "10", "5", "0", "-5"]
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
for a_, key, title in [(ax[0], "noise_clean_training", "Trained on clean words"),
                       (ax[1], "noise_multicondition", "Trained on clean and noisy words")]:
    for n, (c, m) in STYLE.items():
        a_.plot(range(5), [res[key][n][s]["mean"] * 100 for s in snrs], color=c, marker=m,
                ms=4.5, lw=1.6, label=n, mec="white", mew=0.5)
    a_.set_xticks(range(5), snrs)
    a_.set_xlabel("Test SNR (dB)")
    a_.set_title(title, fontsize=10)
    a_.axhline(100 * res["baselines"]["Majority class"]["mean"], color="#8a8984", lw=0.8, ls=":")
    a_.yaxis.grid(True, color="#e6e5e1", lw=0.6)
    a_.set_axisbelow(True)
ax[0].set_ylabel("Accuracy (%)")
ax[0].set_ylim(40, 102)
ax[0].text(0, 100 * res["baselines"]["Majority class"]["mean"] - 1, "majority class", fontsize=8,
           color="#52514e", va="top")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.21), fontsize=9)
fig.savefig(os.path.join(OUT, "fig4_noise.pdf"))
plt.close(fig)

# Figure 5: learning curve
sizes = ["1", "2", "3", "5", "10", "20"]
fig, ax = plt.subplots(figsize=(4.8, 2.9))
for n, (c, m) in STYLE.items():
    ax.plot(range(6), [res["learning_curve"][n][s]["mean"] * 100 for s in sizes], color=c, marker=m,
            ms=4.5, lw=1.6, label=n, mec="white", mew=0.5)
ax.set_xticks(range(6), sizes)
ax.set_xlabel("Training recordings (8 words each)")
ax.set_ylabel("Accuracy (%)")
ax.yaxis.grid(True, color="#e6e5e1", lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.savefig(os.path.join(OUT, "fig5_learning_curve.pdf"))
plt.close(fig)
print("figures written to", OUT)
