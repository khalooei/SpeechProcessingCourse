"""Yes/No word classification: train and compare classifiers on waves_yesno.

    python run_experiments.py --data ../data

Every split is made per recording, never per word. Words from one file share a
session, so a word-level split would let the test set leak into training.
"""
import argparse
import glob
import json
import os
import time
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import audio_utils as au

warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEED = 42
SNRS = [20, 10, 5, 0, -5]
TRAIN_SNRS = [20, 10, 5, 0]


def build_models():
    """Each entry is (pipeline, grid). Grids stay small: the data set is small."""
    return {
        "SVM (RBF)": (make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                      {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.001]}),
        "Neural network (MLP)": (make_pipeline(StandardScaler(), MLPClassifier(
            max_iter=600, random_state=SEED)),
            {"mlpclassifier__hidden_layer_sizes": [(32,), (64, 32)],
             "mlpclassifier__alpha": [1e-4, 1e-2]}),
        "Logistic regression": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                                {"logisticregression__C": [0.01, 0.1, 1, 10]}),
        "Random forest": (make_pipeline(RandomForestClassifier(n_estimators=200, random_state=SEED)),
                          {"randomforestclassifier__max_features": ["sqrt", 0.3]}),
        "k-NN": (make_pipeline(StandardScaler(), KNeighborsClassifier()),
                 {"kneighborsclassifier__n_neighbors": [1, 3, 5, 9]}),
        "Gaussian naive Bayes": (make_pipeline(StandardScaler(), GaussianNB()), {}),
    }


def load_words(folder):
    """Segment every file. Files without 8 clear words are reported and skipped."""
    words, labels, groups, spans, files, skipped = [], [], [], [], [], []
    paths = sorted(glob.glob(os.path.join(folder, "*.wav")))
    for g, path in enumerate(paths):
        x, _ = au.load_wav(path)
        seg = au.segment_words(x)
        if seg is None:
            skipped.append(os.path.basename(path))
            continue
        for (a, b), lab in zip(seg, au.labels_from_name(path)):
            words.append(x[a:b]); labels.append(lab); groups.append(g)
            spans.append((a, b)); files.append(os.path.basename(path))
    return words, np.array(labels), np.array(groups), spans, files, skipped, paths


def fit_best(model, grid, X, y, groups):
    if not grid:
        return clone(model).fit(X, y)
    search = GridSearchCV(model, grid, cv=GroupKFold(n_splits=3), n_jobs=1)
    return search.fit(X, y, groups=groups)


def summary(model_scores):
    a = np.array(model_scores)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "min": float(a.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    folder = au.ensure_dataset(args.data)
    words, y, groups, spans, files, skipped, paths = load_words(folder)
    print(f"{len(words)} words from {len(set(groups))} files; skipped: {skipped}")
    res = {"n_words": len(words), "n_files": len(set(groups.tolist())), "skipped": skipped,
           "n_yes": int(y.sum()), "n_no": int((1 - y).sum()),
           "word_len_s": {"yes": [float(np.mean([len(w) for w, l in zip(words, y) if l == 1]) / 8000)],
                          "no": [float(np.mean([len(w) for w, l in zip(words, y) if l == 0]) / 8000)]}}

    X = np.array([au.word_features(w) for w in words])
    X_simple = np.array([au.simple_features(w) for w in words])
    rng = np.random.default_rng(SEED)
    X_noisy = {s: np.array([au.word_features(au.add_noise(w, s, rng)) for w in words]) for s in SNRS}
    models = build_models()

    # --- A. tutorial-style split: the first 31 sorted file names train, the rest test
    train_names = {os.path.basename(p) for p in paths[:31]}
    is_train = np.array([f in train_names for f in files])
    res["holdout"] = {"train_files": int(len(set(groups[is_train].tolist()))),
                      "test_files": int(len(set(groups[~is_train].tolist()))),
                      "test_words": int((~is_train).sum()),
                      "train_yes_share": float(y[is_train].mean()),
                      "test_yes_share": float(y[~is_train].mean()), "models": {}}
    for name, (m, grid) in models.items():
        est = fit_best(m, grid, X[is_train], y[is_train], groups[is_train])
        p = est.predict(X[~is_train])
        res["holdout"]["models"][name] = {"accuracy": accuracy_score(y[~is_train], p),
                                          "errors": int((p != y[~is_train]).sum())}
        print("holdout", name, res["holdout"]["models"][name])

    # --- B. repeated 5-fold CV, grouped by file, tuning inside each training fold
    cv_acc = {n: [] for n in models}; cv_f1 = {n: [] for n in models}
    cv_fit = {n: [] for n in models}; cv_pred = {n: [] for n in models}
    conf = {n: np.zeros((2, 2), int) for n in models}
    base = {"Majority class": [], "Duration, ZCR and RMS only (SVM)": []}
    wrong = {}
    noise_clean = {n: {s: [] for s in SNRS} for n in models}
    noise_multi = {n: {s: [] for s in SNRS} for n in models}
    chosen = {n: {} for n in models}
    for rep in range(5):
        folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
        for tr, te in folds.split(X, y, groups):
            base["Majority class"].append(accuracy_score(
                y[te], DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr]).predict(X[te])))
            s = make_pipeline(StandardScaler(), SVC()).fit(X_simple[tr], y[tr])
            base["Duration, ZCR and RMS only (SVM)"].append(accuracy_score(y[te], s.predict(X_simple[te])))
            for name, (m, grid) in models.items():
                t0 = time.perf_counter()
                est = fit_best(m, grid, X[tr], y[tr], groups[tr])
                cv_fit[name].append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                p = est.predict(X[te])
                cv_pred[name].append((time.perf_counter() - t0) / len(te) * 1000)
                cv_acc[name].append(accuracy_score(y[te], p))
                cv_f1[name].append(f1_score(y[te], p))
                conf[name] += confusion_matrix(y[te], p, labels=[0, 1])
                for i in te[p != y[te]]:
                    key = f"{files[i]}#{np.where(np.array(files) == files[i])[0].tolist().index(int(i)) + 1}"
                    wrong.setdefault(key, []).append(name)
                if hasattr(est, "best_params_"):
                    k = json.dumps(est.best_params_, sort_keys=True)
                    chosen[name][k] = chosen[name].get(k, 0) + 1
                if rep == 0:  # noise only on the first repeat, to keep run time sensible
                    fixed = est.best_estimator_ if hasattr(est, "best_estimator_") else est
                    Xm = np.vstack([X[tr]] + [X_noisy[s][tr] for s in TRAIN_SNRS])
                    ym = np.tile(y[tr], len(TRAIN_SNRS) + 1)
                    multi = clone(fixed).fit(Xm, ym)
                    for snr in SNRS:
                        noise_clean[name][snr].append(accuracy_score(y[te], fixed.predict(X_noisy[snr][te])))
                        noise_multi[name][snr].append(accuracy_score(y[te], multi.predict(X_noisy[snr][te])))
            print("rep", rep, "fold done", flush=True)

    res["cv"] = {n: {"accuracy": summary(cv_acc[n]), "f1": summary(cv_f1[n]),
                     "fit_s": float(np.mean(cv_fit[n])), "predict_ms_per_word": float(np.mean(cv_pred[n])),
                     "confusion_no_yes": conf[n].tolist(), "chosen_params": chosen[n]} for n in models}
    res["baselines"] = {k: summary(v) for k, v in base.items()}
    res["wrong_words"] = dict(sorted(wrong.items(), key=lambda kv: -len(kv[1])))
    res["noise_clean_training"] = {n: {str(s): summary(v) for s, v in d.items()} for n, d in noise_clean.items()}
    res["noise_multicondition"] = {n: {str(s): summary(v) for s, v in d.items()} for n, d in noise_multi.items()}

    # --- C. learning curve: how few recordings are enough?
    sizes = [1, 2, 3, 5, 10, 20]
    uniq = np.unique(groups)
    lc = {n: {k: [] for k in sizes} for n in models}
    rng2 = np.random.default_rng(SEED)
    for r in range(20):
        perm = rng2.permutation(uniq)
        test_files, pool = perm[:15], perm[15:]
        te = np.isin(groups, test_files)
        for k in sizes:
            tr = np.isin(groups, pool[:k])
            if len(set(y[tr])) < 2:
                continue
            for name, (m, grid) in models.items():
                params = max(chosen[name], key=chosen[name].get) if chosen[name] else "{}"
                est = clone(m).set_params(**{a: (tuple(b) if isinstance(b, list) else b)
                                             for a, b in json.loads(params).items()})
                if name == "k-NN":  # k cannot exceed the number of training words
                    est.set_params(kneighborsclassifier__n_neighbors=min(
                        est.get_params()["kneighborsclassifier__n_neighbors"], int(tr.sum())))
                lc[name][k].append(accuracy_score(y[te], est.fit(X[tr], y[tr]).predict(X[te])))
    res["learning_curve"] = {n: {str(k): summary(v) for k, v in d.items()} for n, d in lc.items()}

    # --- D. a look at the feature space
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(StandardScaler().fit_transform(X))
    res["pca_variance"] = pca.explained_variance_ratio_.tolist()
    np.savez(os.path.join(args.out, "pca.npz"), Z=Z, y=y)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("saved")


if __name__ == "__main__":
    main()
