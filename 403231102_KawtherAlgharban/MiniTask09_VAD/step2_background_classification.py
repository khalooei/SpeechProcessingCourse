"""
Mini-Task 9 - Step 2: Events & Background classification (VOiCES)
Digital Speech Processing - Dr. Mohammad Khalooei - AUT
Student: Kawther Algharban (403231102)

Goal: recognise WHICH background sound was playing during each recording
      none (no distractor) | babb (babble) | musi (music) | tele (television)

Idea: the foreground speech is the same LibriSpeech sentence in every
condition, so the speech itself carries no information about the class.
The background is heard best in the PAUSES between words. We therefore
compare three feature sets (mean + std of 19 frame features each):
   A. whole recording        (all frames)
   B. pauses only            (25% lowest-energy frames = background-dominated)
   C. A + B combined

Evaluation: 5-fold GroupKFold grouped by SENTENCE, so the same sentence is
never in both train and test (otherwise the model could memorise the speech).
Six classifiers: SVM, Neural Network (MLP), k-NN, Random Forest,
Logistic Regression, Gaussian Naive Bayes.

Output:
  results/step2_background_results.csv
  figures/step2_feature_set_comparison.png
  figures/step2_confusion_best.png
  figures/step2_accuracy_by_mic_and_room.png
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from audio_features import (list_files, load_wav, frame_features, CLASSES, CLASS_NAMES,
                            FIG_DIR, RES_DIR, FEAT_DIR)

SEED = 42
PAUSE_PERCENT = 25


def stats(F):
    return np.concatenate([F.mean(axis=0), F.std(axis=0)])


def extract(files):
    cache = FEAT_DIR / "step2_file_features.npz"
    if cache.exists():
        d = np.load(cache)
        if len(d["A"]) == len(files):
            print("Loaded cached features:", cache.name)
            return d["A"], d["B"]
    A, B = [], []
    t0 = time.time()
    for i, f in enumerate(files):
        x, _ = load_wav(f["path"])
        F = frame_features(x)
        energy = F[:, 13]
        pause = F[energy <= np.percentile(energy, PAUSE_PERCENT)]
        A.append(stats(F))
        B.append(stats(pause))
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            print(f"   features: {i + 1}/{len(files)} files  ({time.time() - t0:.0f} s)", flush=True)
    A, B = np.array(A), np.array(B)
    FEAT_DIR.mkdir(exist_ok=True)
    np.savez(cache, A=A, B=B)
    return A, B


def build_models():
    return {
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale"),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=3000,
                                              early_stopping=True, random_state=SEED),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Gaussian Naive Bayes": GaussianNB(),
    }


def main():
    files = list_files()
    y = np.array([CLASSES.index(f["distractor"]) for f in files])
    groups = np.array([f["sentence"] for f in files])
    mic = np.array([f["mic_loc"] for f in files])
    room = np.array([f["room"] for f in files])
    print(f"Files: {len(files)} | sentences: {len(set(groups))} | classes:",
          {c: int((y == i).sum()) for i, c in enumerate(CLASSES)})

    A, B = extract(files)
    sets = {"A: whole recording": A, "B: pauses only": B, "C: whole + pauses": np.hstack([A, B])}

    cv = GroupKFold(n_splits=5)
    rows, preds = [], {}
    print(f"\n{'Feature set':<22}{'Model':<24}{'Accuracy':>10}{'Macro-F1':>10}{'Time s':>8}")
    print("-" * 74)
    for set_name, X in sets.items():
        for name, clf in build_models().items():
            model = make_pipeline(StandardScaler(), clf)
            t0 = time.time()
            p = cross_val_predict(model, X, y, groups=groups, cv=cv)
            dt = time.time() - t0
            acc, f1 = accuracy_score(y, p), f1_score(y, p, average="macro")
            rows.append((set_name, name, acc, f1, dt))
            preds[(set_name, name)] = p
            print(f"{set_name:<22}{name:<24}{acc * 100:>9.2f}%{f1:>10.3f}{dt:>8.1f}", flush=True)

    best = max(rows, key=lambda r: r[2])
    best_set, best_model = best[0], best[1]
    p_best = preds[(best_set, best_model)]
    print(f"\nBest: {best_model} on '{best_set}'  ->  accuracy {best[2] * 100:.2f}%")
    print(classification_report(y, p_best, target_names=[CLASS_NAMES[c] for c in CLASSES], digits=3))

    print("Accuracy of the best model by microphone position and room:")
    for key, arr in [("mic", mic), ("room", room)]:
        for v in sorted(set(arr)):
            m = arr == v
            print(f"   {key} {v:<5}: {accuracy_score(y[m], p_best[m]) * 100:6.2f}%  ({m.sum()} files)")

    # ---------- save table ----------
    RES_DIR.mkdir(exist_ok=True)
    with open(RES_DIR / "step2_background_results.csv", "w") as f:
        f.write("feature_set,model,accuracy,macro_f1,time_s\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.2f}\n")

    FIG_DIR.mkdir(exist_ok=True)
    # ---------- Figure 1: feature-set comparison ----------
    models = list(build_models().keys())
    pos = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(13, 5))
    for k, s in enumerate(sets):
        vals = [next(r[2] for r in rows if r[0] == s and r[1] == m) * 100 for m in models]
        ax.bar(pos + (k - 1) * 0.27, vals, 0.27, label=s)
        for i, v in enumerate(vals):
            ax.text(pos[i] + (k - 1) * 0.27, v + 0.5, f"{v:.0f}", ha="center", fontsize=7)
    ax.axhline(25, color="gray", ls="--", lw=1)
    ax.text(len(models) - 0.5, 26, "chance (25%)", color="gray", fontsize=8, ha="right")
    ax.set_xticks(pos)
    ax.set_xticklabels(models, rotation=12)
    ax.set_ylabel("Accuracy (%)  - 5-fold grouped CV")
    ax.set_ylim(0, 105)
    ax.set_title("Background classification (none / babble / music / TV): feature sets x classifiers")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step2_feature_set_comparison.png", dpi=150)

    # ---------- Figure 2: confusion matrix of the best model ----------
    cm = confusion_matrix(y, p_best)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    labels = [CLASS_NAMES[c] for c in CLASSES]
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_yticks(range(4))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{best_model}\n{best_set} - accuracy {best[2] * 100:.1f}%", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step2_confusion_best.png", dpi=150)

    # ---------- Figure 3: accuracy by microphone / room ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    order = [m for m in ["clo", "mid", "far", "beh"] if m in set(mic)]
    names = {"clo": "close", "mid": "middle", "far": "far", "beh": "behind"}
    ax[0].bar([names[m] for m in order],
              [accuracy_score(y[mic == m], p_best[mic == m]) * 100 for m in order], color="tab:blue")
    ax[0].set_title("Accuracy by microphone position")
    rooms = sorted(set(room))
    ax[1].bar(rooms, [accuracy_score(y[room == r], p_best[room == r]) * 100 for r in rooms],
              color="tab:orange")
    ax[1].set_title("Accuracy by room")
    for a in ax:
        a.set_ylim(0, 105)
        a.set_ylabel("Accuracy (%)")
        for b in a.patches:
            a.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}",
                   ha="center", fontsize=9)
    fig.suptitle(f"Best model ({best_model}, {best_set})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step2_accuracy_by_mic_and_room.png", dpi=150)
    print("Figures saved to:", FIG_DIR)
    plt.show()


if __name__ == "__main__":
    main()
