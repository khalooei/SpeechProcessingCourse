"""Turn results.json into LaTeX tables and number macros, so the report cannot drift from the run."""
import json
import os
import sys

RES = sys.argv[1] if len(sys.argv) > 1 else "../results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "../report/generated.tex"
r = json.load(open(os.path.join(RES, "results.json")))
NAMES = list(r["cv"])
p = lambda v, d=2: f"{v * 100:.{d}f}"

REPORT = os.path.dirname(OUT)
t = [r"\begin{table}[H]\centering\small",
     r"\caption{Clean test results. CV columns: 5-fold cross-validation repeated 5 times, split by recording (25 folds, 2360 test predictions in all). Hold-out: 30 training and 29 test recordings (232 test words).}\label{tab:clean}",
     r"\setlength{\tabcolsep}{4pt}\begin{tabular}{lccccc}\toprule",
     r"Classifier & Hold-out acc. (\%) & Hold-out errors & CV acc. (\%) & CV F1 (\%) & CV errors\\\midrule"]
for n in NAMES:
    c = r["cv"][n]; h = r["holdout"]["models"][n]
    errs = c["confusion_no_yes"][0][1] + c["confusion_no_yes"][1][0]
    t.append(f"{n} & {p(h['accuracy'])} & {h['errors']} & {p(c['accuracy']['mean'])} $\\pm$ {p(c['accuracy']['std'])} & {p(c['f1']['mean'])} & {errs}\\\\")
t.append(r"\midrule")
for n, v in r["baselines"].items():
    t.append(f"{n.replace('Duration, ZCR and RMS only (SVM)', 'Time-domain features (SVM)')} & -- & -- & {p(v['mean'])} $\\pm$ {p(v['std'])} & -- & --\\\\")
t += [r"\bottomrule\end{tabular}\end{table}"]
open(os.path.join(REPORT, "tables_clean.tex"), "w").write("\n".join(t) + "\n")

t = [r"\begin{table}[H]\centering\small",
      r"\caption{Accuracy (\%) on noisy test words, mean $\pm$ std over 5 folds.}\label{tab:noise}",
      r"\setlength{\tabcolsep}{2.5pt}\footnotesize\begin{tabular}{lcccccc}\toprule",
      r"& \multicolumn{3}{c}{Trained on clean words} & \multicolumn{3}{c}{Trained on clean and noisy words}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
      r"Classifier & 10 dB & 5 dB & 0 dB & 10 dB & 5 dB & 0 dB\\\midrule"]
for n in NAMES:
    row = [n]
    for key in ["noise_clean_training", "noise_multicondition"]:
        for s in ["10", "5", "0"]:
            v = r[key][n][s]
            row.append(f"{v['mean'] * 100:.1f} $\\pm$ {v['std'] * 100:.1f}")
    t.append(" & ".join(row) + r"\\")
t += [r"\bottomrule\end{tabular}\end{table}"]
open(os.path.join(REPORT, "tables_noise.tex"), "w").write("\n".join(t) + "\n")

t = [r"\begin{table}[H]\centering\small",
      r"\caption{Average cost per training fold and per word on the 2-core machine used here. Fit time includes the grid search, and naive Bayes has no grid, so its figure is not directly comparable.}\label{tab:cost}",
      r"\begin{tabular}{lcc}\toprule Classifier & Fit with tuning (s) & Prediction (ms/word)\\\midrule"]
for n in NAMES:
    t.append(f"{n} & {r['cv'][n]['fit_s']:.2f} & {r['cv'][n]['predict_ms_per_word']:.3f}\\\\")
t += [r"\bottomrule\end{tabular}\end{table}"]
open(os.path.join(REPORT, "tables_cost.tex"), "w").write("\n".join(t) + "\n")

t = [r"\begin{table}[H]\centering\small",
     r"\caption{Cross-validation precision and recall (\%) with ``yes'' as the positive class, and the confusion counts summed over all 25 test folds.}\label{tab:pr}",
     r"\begin{tabular}{lcccccc}\toprule Classifier & Precision & Recall & True no & False yes & False no & True yes\\\midrule"]
for n in NAMES:
    (tn, fp), (fn, tp) = r["cv"][n]["confusion_no_yes"]
    t.append(f"{n} & {100 * tp / max(tp + fp, 1):.2f} & {100 * tp / max(tp + fn, 1):.2f} & {tn} & {fp} & {fn} & {tp}\\\\")
t += [r"\bottomrule\end{tabular}\end{table}"]
open(os.path.join(REPORT, "tables_pr.tex"), "w").write("\n".join(t) + "\n")

f = json.load(open(os.path.join(RES, "results_filterbank.json")))
t = [r"\begin{table}[H]\centering\small",
     r"\caption{Accuracy (\%) with 26 log filter bank energies (FB) and with MFCCs, on the same splits and the same noise.}\label{tab:fbank}",
     r"\setlength{\tabcolsep}{4pt}\begin{tabular}{lcccccc}\toprule",
     r"& \multicolumn{2}{c}{Clean words} & \multicolumn{2}{c}{10 dB, clean training} & \multicolumn{2}{c}{0 dB, multi-condition}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
     r"Classifier & FB & MFCC & FB & MFCC & FB & MFCC\\\midrule"]
for n in NAMES:
    t.append(f"{n} & {f['clean_cv'][n]['mean'] * 100:.2f} & {r['cv'][n]['accuracy']['mean'] * 100:.2f} & "
             f"{f['noise_clean_training'][n]['10']['mean'] * 100:.1f} & {r['noise_clean_training'][n]['10']['mean'] * 100:.1f} & "
             f"{f['noise_multicondition'][n]['0']['mean'] * 100:.1f} & {r['noise_multicondition'][n]['0']['mean'] * 100:.1f}\\\\")
t += [r"\bottomrule\end{tabular}\end{table}"]
open(os.path.join(REPORT, "tables_fbank.tex"), "w").write("\n".join(t) + "\n")

m = {"NWords": r["n_words"], "NFiles": r["n_files"], "NYes": r["n_yes"], "NNo": r["n_no"],
     "TimeBase": p(r["baselines"]["Duration, ZCR and RMS only (SVM)"]["mean"], 1),
     "Major": p(r["baselines"]["Majority class"]["mean"], 1),
     "LenYes": f"{r['word_len_s']['yes'][0]:.2f}", "LenNo": f"{r['word_len_s']['no'][0]:.2f}",
     "PCAone": f"{r['pca_variance'][0] * 100:.0f}", "TestYes": p(r["holdout"]["test_yes_share"], 0), "TrainYes": p(r["holdout"]["train_yes_share"], 0),
     "LRcleanZero": f"{r['noise_clean_training']['Logistic regression']['0']['mean'] * 100:.1f}",
     "SVMcleanZero": f"{r['noise_clean_training']['SVM (RBF)']['0']['mean'] * 100:.1f}",
     "MLPcleanZero": f"{r['noise_clean_training']['Neural network (MLP)']['0']['mean'] * 100:.1f}",
     "MLPmultiZero": f"{r['noise_multicondition']['Neural network (MLP)']['0']['mean'] * 100:.1f}",
     "SVMmultiZero": f"{r['noise_multicondition']['SVM (RBF)']['0']['mean'] * 100:.1f}",
     "SVMmultiMinus": f"{r['noise_multicondition']['SVM (RBF)']['-5']['mean'] * 100:.1f}",
     "MLPmultiMinus": f"{r['noise_multicondition']['Neural network (MLP)']['-5']['mean'] * 100:.1f}",
     "NBmultiZero": f"{r['noise_multicondition']['Gaussian naive Bayes']['0']['mean'] * 100:.1f}",
     "KNNmultiZero": f"{r['noise_multicondition']['k-NN']['0']['mean'] * 100:.1f}",
     "SVMlcOne": f"{r['learning_curve']['SVM (RBF)']['1']['mean'] * 100:.1f}",
     "SVMlcThree": f"{r['learning_curve']['SVM (RBF)']['3']['mean'] * 100:.1f}",
     "SVMlcFive": f"{r['learning_curve']['SVM (RBF)']['5']['mean'] * 100:.1f}",
     "KNNlcOne": f"{r['learning_curve']['k-NN']['1']['mean'] * 100:.1f}",
     "MLPlcOne": f"{r['learning_curve']['Neural network (MLP)']['1']['mean'] * 100:.1f}"}
t = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
open(OUT, "w").write("\n".join(t) + "\n")
print("wrote", OUT)
