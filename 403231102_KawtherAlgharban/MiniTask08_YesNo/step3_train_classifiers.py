"""
Mini-Task 8 - Step 3: Train and compare classifiers (YES vs NO)
Digital Speech Processing - Dr. Mohammad Khalooei - AUT
Student: Kawther Algharban (403231102)

Input : features/yesno_features.npz   (made by step 2)
        X = 52 MFCC statistics per word, y = 1 (yes) / 0 (no), file_id

Evaluation protocol
  * Words from the SAME recording are never split between train and test
    (GroupKFold / GroupShuffleSplit on file_id) -> no information leakage.
  * 5-fold grouped cross-validation  -> mean +- std accuracy
  * One fixed hold-out split (70% train files / 30% test files)
    -> precision, recall, F1, confusion matrix

Classifiers
  1. SVM (RBF kernel)
  2. Neural Network (Multi-Layer Perceptron)
  3. k-Nearest Neighbours
  4. Random Forest
  5. Logistic Regression
  6. Gaussian Naive Bayes

Output
  results/results_table.csv
  figures/step3_accuracy_comparison.png
  figures/step3_confusion_matrices.png
"""

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

BASE_DIR = Path(__file__).resolve().parent
FEAT_FILE = BASE_DIR / "features" / "yesno_features.npz"
FIG_DIR = BASE_DIR / "figures"
RES_DIR = BASE_DIR / "results"
SEED = 42


def build_models():
    """Every model is a pipeline: standardisation -> classifier."""
    return {
        "SVM (RBF)": make_pipeline(StandardScaler(),
                                   SVC(kernel="rbf", C=10, gamma="scale")),
        "Neural Network (MLP)": make_pipeline(StandardScaler(),
                                              MLPClassifier(hidden_layer_sizes=(64, 32),
                                                            max_iter=2000,
                                                            random_state=SEED)),
        "k-NN (k=5)": make_pipeline(StandardScaler(),
                                    KNeighborsClassifier(n_neighbors=5)),
        "Random Forest": make_pipeline(StandardScaler(),
                                       RandomForestClassifier(n_estimators=200,
                                                              random_state=SEED)),
        "Logistic Regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=2000)),
        "Gaussian Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
    }


def main():
    if not FEAT_FILE.exists():
        raise SystemExit("Features not found. Run step 2 first.")
    d = np.load(FEAT_FILE)
    X, y, groups = d["X"], d["y"], d["file_id"]
    print(f"Data: {X.shape[0]} words, {X.shape[1]} features, "
          f"{len(np.unique(groups))} recordings  (yes = {y.sum()}, no = {len(y) - y.sum()})")

    # ---------- fixed hold-out split by recording ----------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    tr, te = next(gss.split(X, y, groups))
    print(f"Hold-out split: train = {len(tr)} words ({len(np.unique(groups[tr]))} files), "
          f"test = {len(te)} words ({len(np.unique(groups[te]))} files)\n")

    cv = GroupKFold(n_splits=5)
    models = build_models()
    rows, cms = [], {}

    header = f"{'Model':<24}{'CV acc (mean±std)':>20}{'Test acc':>10}{'Prec':>7}{'Recall':>8}{'F1':>7}{'Train s':>9}"
    print(header)
    print("-" * len(header))

    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring="accuracy")

        t0 = time.perf_counter()
        model.fit(X[tr], y[tr])
        train_time = time.perf_counter() - t0
        pred = model.predict(X[te])

        r = {
            "model": name,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_acc": accuracy_score(y[te], pred),
            "precision": precision_score(y[te], pred, zero_division=0),
            "recall": recall_score(y[te], pred, zero_division=0),
            "f1": f1_score(y[te], pred, zero_division=0),
            "train_time_s": train_time,
        }
        rows.append(r)
        cms[name] = confusion_matrix(y[te], pred, labels=[0, 1])
        print(f"{name:<24}{r['cv_mean']*100:>12.2f} ± {r['cv_std']*100:5.2f}"
              f"{r['test_acc']*100:>9.2f}%{r['precision']:>7.3f}{r['recall']:>8.3f}"
              f"{r['f1']:>7.3f}{train_time:>9.3f}")

    # ---------- best model: detailed report ----------
    best = max(rows, key=lambda r: (r["cv_mean"], -r["cv_std"]))
    print(f"\nBest model by cross-validation: {best['model']}")
    best_model = models[best["model"]]
    print(classification_report(y[te], best_model.predict(X[te]),
                                target_names=["NO (lo)", "YES (ken)"], digits=3))

    # ---------- save table ----------
    RES_DIR.mkdir(exist_ok=True)
    keys = list(rows[0].keys())
    with open(RES_DIR / "results_table.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(round(r[k], 4)) if isinstance(r[k], float) else r[k]
                             for k in keys) + "\n")
    print("Table saved to:", RES_DIR / "results_table.csv")

    # ---------- Figure 1: accuracy comparison ----------
    FIG_DIR.mkdir(exist_ok=True)
    names = [r["model"] for r in rows]
    cv_m = [r["cv_mean"] * 100 for r in rows]
    cv_s = [r["cv_std"] * 100 for r in rows]
    test = [r["test_acc"] * 100 for r in rows]
    pos = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(pos - 0.2, cv_m, 0.4, yerr=cv_s, capsize=4, label="5-fold grouped CV (mean ± std)")
    ax.bar(pos + 0.2, test, 0.4, label="Hold-out test")
    for i in range(len(names)):
        ax.text(pos[i] - 0.2, cv_m[i] + cv_s[i] + 0.3, f"{cv_m[i]:.1f}", ha="center", fontsize=8)
        ax.text(pos[i] + 0.2, test[i] + 0.3, f"{test[i]:.1f}", ha="center", fontsize=8)
    ax.set_xticks(pos)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(max(0, min(cv_m + test) - 10), 101.5)
    ax.set_title("YES / NO word classification - classifier comparison (MFCC features)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step3_accuracy_comparison.png", dpi=150)

    # ---------- Figure 2: confusion matrices ----------
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axk, name in zip(axes.ravel(), names):
        cm = cms[name]
        axk.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                axk.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        axk.set_xticks([0, 1])
        axk.set_xticklabels(["NO", "YES"])
        axk.set_yticks([0, 1])
        axk.set_yticklabels(["NO", "YES"])
        axk.set_xlabel("Predicted")
        axk.set_ylabel("True")
        axk.set_title(name, fontsize=10)
    fig.suptitle("Confusion matrices on the hold-out test set")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step3_confusion_matrices.png", dpi=150)
    print("Figures saved to:", FIG_DIR)
    plt.show()


if __name__ == "__main__":
    main()
