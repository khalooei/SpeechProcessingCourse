"""Mini Task 9 on the VOiCES 90-degree subset (github.com/Lab41/VOiCES-subset).

    python run_voices_subset.py --audio <folder with the 908 wav files> --out ../results

  A. Background classification: none / babble / music / television, one label per file.
  B. Speech / non-speech: 0.5 s windows that are clearly speech or clearly a pause.
  C. Voice activity detection: one decision per 10 ms frame.

Every split is grouped by sentence (speaker-chapter-segment), so a sentence used in
training never appears in testing, in any room, with any distractor or microphone.
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import vad_utils as vu
import voices_subset as vs

SEED = 42
DIST = ["none", "babb", "musi", "tele"]
NAMES = {"none": "none", "babb": "babble", "musi": "music", "tele": "television"}


def file_models():
    return {
        "SVM (RBF)": (make_pipeline(StandardScaler(), SVC()), {"svc__C": [1, 10, 100]}),
        "Neural network (MLP)": (make_pipeline(StandardScaler(), MLPClassifier(max_iter=1500, random_state=SEED)),
                                 {"mlpclassifier__hidden_layer_sizes": [(64,), (128, 64)],
                                  "mlpclassifier__alpha": [1e-3, 1e-1]}),
        "Logistic regression": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)),
                                {"logisticregression__C": [0.1, 1, 10]}),
        "Random forest": (RandomForestClassifier(300, random_state=SEED), {"max_features": ["sqrt", 0.3]}),
        "k-NN": (make_pipeline(StandardScaler(), KNeighborsClassifier()), {"kneighborsclassifier__n_neighbors": [3, 7, 15]}),
        "Gaussian naive Bayes": (make_pipeline(StandardScaler(), GaussianNB()), {}),
    }


def level_features(x):
    """How the band energy is spread over the file, relative to its loud frames.

    A distractor raises the quiet frames, and different distractors fill them in
    different ways, so these percentiles describe the background without a VAD.
    """
    e = vs.band_env(x)
    top = np.percentile(e, 98)
    q = np.percentile(e, [2, 5, 10, 25, 50, 75]) - top
    d = np.abs(np.diff(e))
    return np.r_[q, np.percentile(e, 90) - np.percentile(e, 10), d.mean(), np.percentile(d, 90)]


def summary(v):
    v = np.asarray(v, float)
    return [float(v.mean()), float(v.std(ddof=1))]


def part_a(rows, X, Xl, y, g):
    out = {}
    sets = {"MFCC statistics": X, "MFCC statistics + energy levels": np.hstack([X, Xl])}
    mic = np.array([r["mic_loc"] for r in rows])
    room = np.array([r["room"] for r in rows])
    for fs, F in sets.items():
        out[fs] = {}
        for name, (m, grid) in file_models().items():
            acc, mf, conf = [], [], np.zeros((4, 4), int)
            right = np.zeros(len(y), bool)
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(F, y, g):
                est = GridSearchCV(m, grid, cv=GroupKFold(3)).fit(F[tr], y[tr], groups=g[tr]) if grid else m.fit(F[tr], y[tr])
                p = est.predict(F[te])
                acc.append(accuracy_score(y[te], p)); mf.append(f1_score(y[te], p, average="macro"))
                conf += confusion_matrix(y[te], p, labels=range(4)); right[te] = p == y[te]
            out[fs][name] = {"accuracy": summary(acc), "macro_f1": summary(mf), "confusion": conf.tolist(),
                             "by_mic": {k: float(right[mic == k].mean()) for k in ["clo", "mid", "far", "beh"]},
                             "by_room": {k: float(right[room == k].mean()) for k in ["rm1", "rm2"]}}
            print("A", fs, name, round(np.mean(acc), 3), flush=True)
    return out


def windows(item, size=50, hop=25):
    """0.5 s windows that are >= 95 % speech (1) or <= 5 % speech (0)."""
    x, y = item["x"], item["y"]
    out = []
    for s in range(0, len(y) - size + 1, hop):
        share = y[s:s + size].mean()
        if share >= 0.95 or share <= 0.05:
            out.append((vu.clip_features(x[s * vu.HOP:(s + size) * vu.HOP + vu.FRAME]), int(share >= 0.95)))
    return out


def tune_rule(rule, items, grid):
    best, best_f = grid[0], -1
    for a in grid:
        c = dict(tp=0, fp=0, fn=0, tn=0)
        for it in items:
            p = rule(it["x"], a); L = min(len(p), len(it["y"]))
            for k, v in vu.frame_scores(it["y"][:L], p[:L]).items():
                c[k] += v
        f = vu.from_counts(c)["f1"]
        if f > best_f:
            best, best_f = a, f
    return float(best)


def add(c, y, p):
    L = min(len(y), len(p))
    for k, v in vu.frame_scores(y[:L], p[:L]).items():
        c[k] += v


def parts_bc(items):
    """Speech/non-speech windows (B) and frame VAD (C), 4 folds grouped by sentence.

    The 'none' recording that produced a group's labels is never a test file: its labels
    came from its own energy, which would favour the energy rule.
    """
    utt = np.array([it["row"]["utt"] for it in items])
    uniq = np.unique(utt)
    rng = np.random.default_rng(SEED)
    fold_of = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % 4))
    B = defaultdict(lambda: {"y": [], "p": [], "d": []})
    C = defaultdict(lambda: defaultdict(lambda: dict(tp=0, fp=0, fn=0, tn=0)))
    alphas = defaultdict(list)
    grid = list(np.round(np.arange(0.1, 0.91, 0.05), 2))
    feats = {}
    for k in range(4):
        tr = [it for it in items if fold_of[it["row"]["utt"]] != k]
        te = [it for it in items if fold_of[it["row"]["utt"]] == k and it["row"]["file"] != it["ref"]]
        # B: windows
        Wtr = [w for it in tr for w in windows(it)]
        Xw, yw = np.array([w[0] for w in Wtr]), np.array([w[1] for w in Wtr])
        wm = {"SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, class_weight="balanced")),
              "Neural network (MLP)": make_pipeline(StandardScaler(), MLPClassifier((64,), alpha=1e-2, max_iter=800, random_state=SEED)),
              "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))}
        for m in wm.values():
            m.fit(Xw, yw)
        for it in te:
            W = windows(it)
            if not W:
                continue
            Xt = np.array([w[0] for w in W]); yt = [w[1] for w in W]
            for n, m in wm.items():
                B[n]["y"] += yt; B[n]["p"] += m.predict(Xt).tolist(); B[n]["d"] += [it["row"]["distractor"]] * len(yt)
        # C: frames
        ae = tune_rule(vu.energy_vad, tr[::3], grid); ah = tune_rule(vu.entropy_vad, tr[::3], grid)
        alphas["energy"].append(ae); alphas["entropy"].append(ah)
        Xf, yf = [], []
        for it in tr:
            key = it["row"]["file"]
            if key not in feats:
                feats[key] = vu.frame_features(it["x"])
            F = feats[key]; L = min(len(F), len(it["y"]))
            pick = rng.choice(L, min(300, L), replace=False)
            Xf.append(F[pick]); yf.append(it["y"][pick])
        Xf, yf = np.vstack(Xf), np.concatenate(yf)
        fm = {"Logistic regression (frames)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
              "Neural network (frames)": make_pipeline(StandardScaler(), MLPClassifier((128, 64), alpha=1e-3, early_stopping=True, max_iter=200, random_state=SEED))}
        for m in fm.values():
            m.fit(Xf, yf)
        for it in te:
            key = it["row"]["file"]
            F = feats.get(key)
            if F is None:
                F = feats[key] = vu.frame_features(it["x"])
            pred = {"Energy threshold": vu.energy_vad(it["x"], ae), "Spectral entropy": vu.entropy_vad(it["x"], ah)}
            for n, m in fm.items():
                pred[n] = vu.smooth(m.predict(F))
            for n, p in pred.items():
                for grp in ["all", "d:" + it["row"]["distractor"], "m:" + it["row"]["mic_loc"]]:
                    add(C[n][grp], it["y"], p)
        print("fold", k, "done", flush=True)
    resB = {}
    for n, d in B.items():
        y, p, dd = np.array(d["y"]), np.array(d["p"]), np.array(d["d"])
        resB[n] = {"windows": int(len(y)), "speech_share": float(y.mean()),
                   "precision": precision_score(y, p), "recall": recall_score(y, p), "f1": f1_score(y, p),
                   "nonspeech_recall": float((p[y == 0] == 0).mean()),
                   "balanced_accuracy": float(0.5 * ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean())),
                   "by_distractor_bal_acc": {k: float(0.5 * ((p[(y == 1) & (dd == k)] == 1).mean() + (p[(y == 0) & (dd == k)] == 0).mean()))
                                             for k in DIST}}
    resC = {n: {g: vu.from_counts(c) for g, c in d.items()} for n, d in C.items()}
    tot = C["Energy threshold"]["all"]
    return resB, {"scores": resC, "alphas": alphas, "speech_share": (tot["tp"] + tot["fn"]) / sum(tot.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    t0 = time.time()
    rows = vs.index(a.audio)
    print(len(rows), "files", flush=True)
    items = vs.labelled_set(rows)
    info = {"files": len(rows), "sentences": len({r["utt"] for r in rows}), "speakers": len({r["speaker"] for r in rows}),
            "by_distractor": Counter(r["distractor"] for r in rows), "by_room": Counter(r["room"] for r in rows),
            "by_mic": Counter(r["mic_loc"] for r in rows),
            "groups": len(vs.groups(rows)), "labelled_files": len(items),
            "groups_with_none": len({(it["row"]["utt"], it["row"]["room"]) for it in items}),
            "align_peak_pct": np.percentile([it["peak"] for it in items if it["row"]["file"] != it["ref"]], [5, 50]).tolist(),
            "minutes": sum(len(it["x"]) for it in items) / vu.SR / 60}
    cache = {it["row"]["file"]: it["x"] for it in items}
    X, Xl, y, g = [], [], [], []
    for r in rows:
        x = cache.get(r["file"])
        if x is None:
            x = vu.load_audio(r["path"])
        X.append(vu.clip_features(x)); Xl.append(level_features(x)); y.append(DIST.index(r["distractor"])); g.append(r["utt"])
    res = {"info": info, "A": part_a(rows, np.array(X), np.array(Xl), np.array(y), np.array(g))}
    res["B"], res["C"] = parts_bc(items)
    res["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "voices_subset_results.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda o: dict(o) if isinstance(o, Counter) else str(o))
    ex = next(it for it in items if it["row"]["distractor"] == "musi" and it["row"]["mic_loc"] == "far")
    np.savez(os.path.join(a.out, "voices_example.npz"), x=ex["x"], y=ex["y"], name=ex["row"]["file"])
    print("saved in", round(res["seconds"]), "s")


if __name__ == "__main__":
    main()
