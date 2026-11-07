"""Figures and LaTeX numbers for the Task 2 preparation note."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "../results"
REPORT = sys.argv[2] if len(sys.argv) > 2 else "../report"
os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "savefig.dpi": 220, "savefig.bbox": "tight", "pdf.fonttype": 42})
NAVY, GOLD = "#14305c", "#c8981e"
r = json.load(open(os.path.join(RES, "results.json")))
S = r["part3"]["scores"]
STY = {"Energy threshold": ("#2a78d6", "o"), "Spectral entropy": ("#eb6834", "s"),
       "Logistic regression (frames)": ("#1baf7a", "^"), "Neural network (frames)": ("#eda100", "D")}
snrs = ["20", "10", "5", "0"]

fig, ax = plt.subplots(figsize=(4.6, 2.7))
for n, (c, m) in STY.items():
    ax.plot(range(4), [S[n][s]["f1"] * 100 for s in snrs], color=c, marker=m, ms=5, lw=1.8,
            mec="white", mew=0.5, label=n)
ax.set_xticks(range(4), snrs)
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Speech F1 (%)"); ax.set_ylim(0, 100)
ax.yaxis.grid(True, color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2, bbox_to_anchor=(0, -0.42))
fig.savefig(os.path.join(RES, "figures", "fig2_vad_f1.pdf")); plt.close(fig)

e = np.load(os.path.join(RES, "vad_example.npz"))
x = e["x"]; t = np.arange(len(x)) / 8000
ft = np.arange(len(e["truth"])) * 0.01
rows = [("truth", "Reference", NAVY), ("Energy_threshold", "Energy", "#2a78d6"),
        ("Neural_network_(frames)", "Neural net", GOLD)]
fig, ax = plt.subplots(2, 1, figsize=(6.6, 2.7), sharex=True, height_ratios=[2, 1.2])
ax[0].plot(t, x, color="#3f4a5c", lw=0.25); ax[0].set_ylabel("Amplitude")
for i, (k, n, c) in enumerate(rows):
    L = min(len(ft), len(e[k]))
    ax[1].fill_between(ft[:L], i + 0.12, i + 0.88, where=e[k][:L].astype(bool), color=c, lw=0, step="post")
ax[1].set_yticks([0.5, 1.5, 2.5], [n for _, n, _ in rows]); ax[1].set_ylim(0, 3)
ax[1].invert_yaxis(); ax[1].set_xlabel("Time (s)"); ax[1].spines["left"].set_visible(False)
ax[1].tick_params(axis="y", length=0)
fig.savefig(os.path.join(RES, "figures", "fig1_vad_example.pdf")); plt.close(fig)

p = lambda v: f"{v * 100:.1f}"
t1 = [r"\begin{table}[H]\centering\small\caption{ESC-10 results, official 5 folds, accuracy (\%) $\pm$ std.}\label{tab:esc}",
      r"\begin{tabular}{lccc}\toprule Classifier & Event vs background & 10 classes & 10-class macro F1\\\midrule"]
for n, d in r["part1"].items():
    t1.append(f"{n} & {p(d['event_vs_background'][0])} $\\pm$ {p(d['event_vs_background'][1])} & "
              f"{p(d['ten_class'][0])} $\\pm$ {p(d['ten_class'][1])} & {p(d['macro_f1'][0])} $\\pm$ {p(d['macro_f1'][1])}\\\\")
t1.append(r"\bottomrule\end{tabular}\end{table}")
P = r["part2"]
t2 = [r"\begin{table}[H]\centering\small\caption{Speech vs non-speech on 1~s clips, unseen speakers and sounds (%d speech, %d non-speech clips).}\label{tab:sns}" % (P["test_clips"]["speech"], P["test_clips"]["nonspeech"]),
      r"\begin{tabular}{lcccc}\toprule SNR & \multicolumn{2}{c}{SVM} & \multicolumn{2}{c}{MLP}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
      r"(dB) & Bal. acc. (\%) & Speech F1 (\%) & Bal. acc. (\%) & Speech F1 (\%)\\\midrule"]
for s in snrs:
    a, b = P[s]["SVM (RBF)"], P[s]["Neural network (MLP)"]
    t2.append(f"{s} & {p(a['balanced_accuracy'])} & {p(a['speech_f1'])} & {p(b['balanced_accuracy'])} & {p(b['speech_f1'])}\\\\")
t2.append(r"\bottomrule\end{tabular}\end{table}")
t3 = [r"\begin{table}[H]\centering\small\caption{Frame-level VAD at 5 dB on 16 unseen streams of 20 s. F1 is pooled over the streams, and the $\pm$ value is the spread of F1 between single streams. Rates in \%.}\label{tab:vad}",
      r"\begin{tabular}{lcccc}\toprule Method & F1 & Precision & Miss rate & False alarm rate\\\midrule"]
for n in STY:
    d = S[n]["5"]
    pr = d["f1"] * (1 - d["miss_rate"]) / max(2 * (1 - d["miss_rate"]) - d["f1"], 1e-9)
    t3.append(f"{n} & {p(d['f1'])} $\\pm$ {p(d['stream_f1_std'])} & {p(pr)} & {p(d['miss_rate'])} & {p(d['false_alarm_rate'])}\\\\")
t3.append(r"\bottomrule\end{tabular}\end{table}")
ABB = {"chainsaw": "Saw", "clock_tick": "Clock", "crackling_fire": "Fire", "crying_baby": "Baby", "dog": "Dog",
       "helicopter": "Heli", "rain": "Rain", "rooster": "Rooster", "sea_waves": "Sea", "sneezing": "Sneeze"}
c = r["part1"]["SVM (RBF)"]
t4 = [r"\begin{table}[H]\centering\small\caption{ESC-10 confusion matrix of the SVM, pooled over the five folds (400 clips). Rows are true classes and columns are predictions.}\label{tab:conf}",
      r"\setlength{\tabcolsep}{3pt}\begin{tabular}{l" + "c" * 10 + r"}\toprule True & " + " & ".join(ABB[k] for k in c["classes"]) + r"\\\midrule"]
for k, row in zip(c["classes"], c["confusion"]):
    t4.append(ABB[k] + " & " + " & ".join(map(str, row)) + r"\\")
t4.append(r"\bottomrule\end{tabular}\end{table}")
prec = lambda f, rc: f * rc / max(2 * rc - f, 1e-9)
t5 = [r"\begin{table}[H]\centering\small\caption{Precision and recall (\%) for the speech class on the same clips.}\label{tab:pr}",
      r"\begin{tabular}{lcccc}\toprule SNR (dB) & SVM precision & SVM recall & MLP precision & MLP recall\\\midrule"]
for s in snrs:
    a, b = P[s]["SVM (RBF)"], P[s]["Neural network (MLP)"]
    t5.append(f"{s} & {p(prec(a['speech_f1'], a['speech_recall']))} & {p(a['speech_recall'])} & "
              f"{p(prec(b['speech_f1'], b['speech_recall']))} & {p(b['speech_recall'])}\\\\")
t5.append(r"\bottomrule\end{tabular}\end{table}")
for i, tab in zip(["1", "1b", "2", "2b", "3"], [t1, t4, t2, t5, t3]):
    open(os.path.join(REPORT, f"tab{i}.tex"), "w").write("\n".join(tab) + "\n")
    print("\n".join(tab))
Q = r["part3"]
macros = {"SpeechShare": f"{Q['speech_share_test'] * 100:.0f}", "NonSpeechAcc": f"{(1 - Q['speech_share_test']) * 100:.0f}",
          "AlphaE": Q["tuned_alpha"]["energy"], "AlphaH": Q["tuned_alpha"]["entropy"]}
open(os.path.join(REPORT, "generated.tex"), "w").write(
    "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
print(macros, P.get("false_alarm_classes_10dB"))
for n in STY:
    print(n, {s: {k: round(v, 3) for k, v in S[n][s].items()} for s in snrs})
