"""Figures, tables and an example for the VOiCES part of the report.

    python make_voices_figures.py --audio <folder with the 908 wav files>
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import vad_utils as vu
import voices_subset as vs

ap = argparse.ArgumentParser()
ap.add_argument("--audio", required=True)
ap.add_argument("--res", default="../results")
ap.add_argument("--report", default="../report")
a = ap.parse_args()
FIG = os.path.join(a.res, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "savefig.dpi": 220, "savefig.bbox": "tight",
                     "pdf.fonttype": 42})
NAVY, GOLD = "#14305c", "#c8981e"
SEQ = LinearSegmentedColormap.from_list("navy", ["#ffffff", "#9fb6d8", "#3f6aa8", NAVY, "#071a36"])
r = json.load(open(os.path.join(a.res, "voices_subset_results.json")))
DIST = ["none", "babb", "musi", "tele"]
LONG = {"none": "None", "babb": "Babble", "musi": "Music", "tele": "Television"}
rows = vs.index(a.audio)
G = vs.groups(rows)

# one sentence, one room, all four distractors, nearest microphone for each
full = sorted(k for k, m in G.items() if set(DIST) <= {x["distractor"] for x in m})
same_mic = [k for k in full if any(all(any(x["distractor"] == d and x["mic_loc"] == loc for x in G[k]) for d in DIST)
                                   for loc in ["clo", "mid", "far", "beh"])]
key = (same_mic or full)[len(same_mic or full) // 2]
locs = [loc for loc in ["clo", "mid", "far", "beh"] if all(any(x["distractor"] == d and x["mic_loc"] == loc for x in G[key]) for d in DIST)]
pick = {d: sorted([x for x in G[key] if x["distractor"] == d and (not locs or x["mic_loc"] == locs[0])],
                  key=lambda x: vs.MIC_ORDER[x["mic_loc"]])[0] for d in DIST}
fig, ax = plt.subplots(4, 1, figsize=(6.6, 6.0), sharex=True)
fig.subplots_adjust(hspace=0.5)
for i, d in enumerate(DIST):
    x = vu.load_audio(pick[d]["path"])
    S = 10 * np.log10(vu.power_spec(x) + 1e-10)
    ax[i].imshow(S.T, origin="lower", aspect="auto", cmap=SEQ, vmin=S.max() - 70, vmax=S.max(),
                 extent=[0, len(S) * 0.01, 0, 4])
    ax[i].set_ylabel("kHz")
    ax[i].set_title(f"{LONG[d]} ({pick[d]['mic_loc']} microphone)", fontsize=9.5, loc="left")
ax[-1].set_xlabel("Time (s)")
fig.savefig(os.path.join(FIG, "fig3_voices_distractors.pdf")); plt.close(fig)

# VAD example: train a frame MLP on other sentences, show one recording
items = vs.labelled_set(rows)
ex = next(it for it in items if it["row"]["distractor"] == "musi" and it["row"]["mic_loc"] == "far" and it["row"]["file"] != it["ref"])
rng = np.random.default_rng(42)
X, y = [], []
for it in items:
    if it["row"]["utt"] == ex["row"]["utt"]:
        continue
    F = vu.frame_features(it["x"]); L = min(len(F), len(it["y"]))
    k = rng.choice(L, min(150, L), replace=False); X.append(F[k]); y.append(it["y"][k])
net = make_pipeline(StandardScaler(), MLPClassifier((128, 64), alpha=1e-3, early_stopping=True, max_iter=200, random_state=42))
net.fit(np.vstack(X), np.concatenate(y))
alpha_e = float(np.median(r["C"]["alphas"]["energy"]))
pred = {"truth": ex["y"], "Energy_threshold": vu.energy_vad(ex["x"], alpha_e),
        "Neural_network_(frames)": vu.smooth(net.predict(vu.frame_features(ex["x"])))}
np.savez(os.path.join(a.res, "voices_vad_example.npz"), x=ex["x"], name=ex["row"]["file"], **pred,
         **{f"dist_{d}": vu.load_audio(pick[d]["path"]) for d in DIST},
         **{f"distname_{d}": pick[d]["file"] for d in DIST})
x = ex["x"]; t = np.arange(len(x)) / vu.SR
lanes = [("truth", "Reference", NAVY), ("Energy_threshold", "Energy", "#2a78d6"), ("Neural_network_(frames)", "Neural net", GOLD)]
fig, ax = plt.subplots(2, 1, figsize=(6.6, 2.7), sharex=True, height_ratios=[2, 1.2])
ax[0].plot(t, x, color="#3f4a5c", lw=0.25); ax[0].set_ylabel("Amplitude")
for i, (k, n, c) in enumerate(lanes):
    m = pred[k].astype(bool); ft = np.arange(len(m)) * 0.01
    ax[1].fill_between(ft, i + 0.12, i + 0.88, where=m, color=c, lw=0, step="post")
ax[1].set_yticks([0.5, 1.5, 2.5], [n for _, n, _ in lanes]); ax[1].set_ylim(0, 3); ax[1].invert_yaxis()
ax[1].set_xlabel("Time (s)"); ax[1].spines["left"].set_visible(False); ax[1].tick_params(axis="y", length=0)
fig.savefig(os.path.join(FIG, "fig4_voices_vad_example.pdf")); plt.close(fig)

# false alarms by distractor
S = r["C"]["scores"]
STY = {"Energy threshold": "#2a78d6", "Spectral entropy": "#eb6834", "Logistic regression (frames)": "#1baf7a",
       "Neural network (frames)": "#eda100"}
fig, ax = plt.subplots(figsize=(6.2, 2.6))
w = 0.2
for i, (n, c) in enumerate(STY.items()):
    ax.bar(np.arange(4) + (i - 1.5) * w, [S[n]["d:" + d]["false_alarm_rate"] * 100 for d in DIST], w * 0.9,
           color=c, label=n.replace(" (frames)", ""))
ax.set_xticks(range(4), [LONG[d] for d in DIST]); ax.set_ylabel("False alarm rate (%)")
ax.yaxis.grid(True, color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.2))
fig.savefig(os.path.join(FIG, "fig5_voices_false_alarms.pdf")); plt.close(fig)

# tables
p = lambda v: f"{v * 100:.1f}"
A = r["A"]
t = [r"\begin{table}[H]\centering\small\caption{Background classification on VOiCES (4 classes, 908 files). Accuracy and macro F1 (\%) over 5 folds grouped by sentence, mean $\pm$ std. Chance is 25\%.}\label{tab:bg}",
     r"\setlength{\tabcolsep}{4pt}\begin{tabular}{lcccc}\toprule & \multicolumn{2}{c}{MFCC statistics} & \multicolumn{2}{c}{+ energy levels}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
     r"Classifier & Accuracy & Macro F1 & Accuracy & Macro F1\\\midrule"]
f1, f2 = A["MFCC statistics"], A["MFCC statistics + energy levels"]
for n in f1:
    t.append(f"{n} & {p(f1[n]['accuracy'][0])} $\\pm$ {p(f1[n]['accuracy'][1])} & {p(f1[n]['macro_f1'][0])} & "
             f"{p(f2[n]['accuracy'][0])} $\\pm$ {p(f2[n]['accuracy'][1])} & {p(f2[n]['macro_f1'][0])}\\\\")
t.append(r"\bottomrule\end{tabular}\end{table}")
open(os.path.join(a.report, "tab_bg.tex"), "w").write("\n".join(t) + "\n")
c = f2["SVM (RBF)"]["confusion"]
t = [r"\begin{table}[H]\centering\small\caption{Confusion matrix of the SVM with both feature sets, summed over the five folds. Rows are the true background.}\label{tab:bgconf}",
     r"\begin{tabular}{lcccc}\toprule True & None & Babble & Music & Television\\\midrule"]
for d, row in zip(DIST, c):
    t.append(LONG[d] + " & " + " & ".join(map(str, row)) + r"\\")
t.append(r"\bottomrule\end{tabular}\end{table}")
open(os.path.join(a.report, "tab_bgconf.tex"), "w").write("\n".join(t) + "\n")
B = r["B"]
t = [r"\begin{table}[H]\centering\small\caption{Speech/non-speech classification of 0.5~s windows (%d test windows, %s\%% speech). Speech is the positive class.}\label{tab:win}" % (B["SVM (RBF)"]["windows"], p(B["SVM (RBF)"]["speech_share"])),
     r"\begin{tabular}{lccccc}\toprule Classifier & Precision & Recall & F1 & Non-speech recall & Balanced acc.\\\midrule"]
for n, v in B.items():
    t.append(f"{n} & {p(v['precision'])} & {p(v['recall'])} & {p(v['f1'])} & {p(v['nonspeech_recall'])} & {p(v['balanced_accuracy'])}\\\\")
t.append(r"\bottomrule\end{tabular}\end{table}")
open(os.path.join(a.report, "tab_win.tex"), "w").write("\n".join(t) + "\n")
t = [r"\begin{table}[H]\centering\small\caption{Frame-level VAD on VOiCES, all test files pooled (4 folds grouped by sentence). Rates in \%.}\label{tab:vadv}",
     r"\begin{tabular}{lcccc}\toprule Method & F1 & Precision & Miss rate & False alarm rate\\\midrule"]
for n in STY:
    d = S[n]["all"]; rc = 1 - d["miss_rate"]; pr = d["f1"] * rc / max(2 * rc - d["f1"], 1e-9)
    t.append(f"{n} & {p(d['f1'])} & {p(pr)} & {p(d['miss_rate'])} & {p(d['false_alarm_rate'])}\\\\")
t.append(r"\bottomrule\end{tabular}\end{table}")
open(os.path.join(a.report, "tab_vadv.tex"), "w").write("\n".join(t) + "\n")
I = r["info"]
mac = {"VFiles": I["files"], "VSent": I["sentences"], "VSpk": I["speakers"], "VGroups": I["groups_with_none"],
       "VLab": I["labelled_files"], "VMin": f"{I['minutes']:.0f}", "VPeak": f"{I['align_peak_pct'][0]:.2f}",
       "VShare": f"{r['C']['speech_share'] * 100:.0f}", "VAlphaE": f"{alpha_e:.2f}",
       "VExample": key[0].replace("-", "--"), "VRoom": key[1], "VMic": pick["none"]["mic_loc"]}
open(os.path.join(a.report, "generated_voices.tex"), "w").write(
    "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()) + "\n")
print(mac)
