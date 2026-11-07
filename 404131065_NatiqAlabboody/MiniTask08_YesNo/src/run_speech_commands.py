"""The same comparison on Google Speech Commands, using only the yes and no folders.

    python run_speech_commands.py --root path/to/speech_commands_v0.02

Uses the official testing_list.txt and validation_list.txt, which keep speakers apart.
Audio is resampled to 8 kHz so that the front-end stays the same as in the main experiment.
"""
import argparse
import csv
import glob
import os
from math import gcd

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

import audio_utils as au
from run_experiments import build_models, fit_best


def load(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64) / 32768.0
    if sr != au.SAMPLE_RATE:
        g = gcd(sr, au.SAMPLE_RATE)
        x = resample_poly(x, au.SAMPLE_RATE // g, sr // g)
    n = au.SAMPLE_RATE
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--max-per-class", type=int, default=0, help="limit training files per class; 0 uses all")
    ap.add_argument("--out", default="speech_commands_results.csv")
    a = ap.parse_args()
    lists = {}
    for name in ["testing", "validation"]:
        with open(os.path.join(a.root, f"{name}_list.txt")) as f:
            lists[name] = {line.strip() for line in f}
    data = {"train": ([], [], []), "test": ([], [], [])}
    rng = np.random.default_rng(42)
    for label, word in enumerate(["no", "yes"]):
        files = sorted(glob.glob(os.path.join(a.root, word, "*.wav")))
        rel = [f"{word}/{os.path.basename(p)}" for p in files]
        train = [p for p, r in zip(files, rel) if r not in lists["testing"] and r not in lists["validation"]]
        test = [p for p, r in zip(files, rel) if r in lists["testing"]]
        if a.max_per_class and len(train) > a.max_per_class:
            train = [train[i] for i in sorted(rng.choice(len(train), a.max_per_class, replace=False))]
        for split, paths in [("train", train), ("test", test)]:
            for p in paths:
                data[split][0].append(au.word_features(load(p)))
                data[split][1].append(label)
                data[split][2].append(os.path.basename(p).split("_")[0])
    Xtr, ytr, gtr = np.array(data["train"][0]), np.array(data["train"][1]), np.array(data["train"][2])
    Xte, yte = np.array(data["test"][0]), np.array(data["test"][1])
    print(f"{len(ytr)} training and {len(yte)} test words")
    rows = [["classifier", "accuracy", "precision_yes", "recall_yes", "weighted_f1", "tn", "fp", "fn", "tp"]]
    for name, (m, grid) in build_models().items():
        p = fit_best(m, grid, Xtr, ytr, gtr).predict(Xte)
        tn, fp, fn, tp = confusion_matrix(yte, p, labels=[0, 1]).ravel()
        rows.append([name, round(accuracy_score(yte, p), 4), round(precision_score(yte, p, zero_division=0), 4),
                     round(recall_score(yte, p, zero_division=0), 4), round(f1_score(yte, p, average="weighted"), 4),
                     tn, fp, fn, tp])
        print(rows[-1])
    with open(a.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)


if __name__ == "__main__":
    main()
