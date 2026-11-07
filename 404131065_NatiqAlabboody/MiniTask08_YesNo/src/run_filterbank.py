"""Log filter bank energies against MFCCs, on the splits of run_experiments.py.

    python run_filterbank.py --data ../data --out ../results
"""
import argparse
import json
import os

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

import audio_utils as au
from run_experiments import SEED, SNRS, TRAIN_SNRS, build_models, fit_best, load_words, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    folder = au.ensure_dataset(a.data)
    words, y, groups, *_ = load_words(folder)
    X = np.array([au.fbank_features(w) for w in words])
    rng = np.random.default_rng(SEED)            # same seed and order: same noise as the MFCC run
    X_noisy = {s: np.array([au.fbank_features(au.add_noise(w, s, rng)) for w in words]) for s in SNRS}
    models = build_models()
    clean = {n: [] for n in models}
    nc = {n: {s: [] for s in SNRS} for n in models}
    nm = {n: {s: [] for s in SNRS} for n in models}
    for rep in range(5):
        folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
        for tr, te in folds.split(X, y, groups):
            for name, (m, grid) in models.items():
                est = fit_best(m, grid, X[tr], y[tr], groups[tr])
                clean[name].append(accuracy_score(y[te], est.predict(X[te])))
                if rep == 0:
                    fixed = est.best_estimator_ if hasattr(est, "best_estimator_") else est
                    Xm = np.vstack([X[tr]] + [X_noisy[s][tr] for s in TRAIN_SNRS])
                    multi = clone(fixed).fit(Xm, np.tile(y[tr], len(TRAIN_SNRS) + 1))
                    for s in SNRS:
                        nc[name][s].append(accuracy_score(y[te], fixed.predict(X_noisy[s][te])))
                        nm[name][s].append(accuracy_score(y[te], multi.predict(X_noisy[s][te])))
        print("repeat", rep, "done", flush=True)
    res = {"dims": X.shape[1], "clean_cv": {n: summary(v) for n, v in clean.items()},
           "noise_clean_training": {n: {str(s): summary(v) for s, v in d.items()} for n, d in nc.items()},
           "noise_multicondition": {n: {str(s): summary(v) for s, v in d.items()} for n, d in nm.items()}}
    with open(os.path.join(a.out, "results_filterbank.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
