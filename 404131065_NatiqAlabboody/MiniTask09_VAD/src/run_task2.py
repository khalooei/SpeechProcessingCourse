"""Rehearsal of the VOiCES task on data available without VOiCES.

  Part 1: events vs background, and 10-class labelling, on ESC-10.
  Part 2: speech vs non-speech on 1 s clips (FSDD speakers vs ESC-50 sounds).
  Part 3: frame-level VAD on simulated distant recordings.

    python run_task2.py --esc path/to/ESC-50 --fsdd path/to/free-spoken-digit-dataset
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import vad_utils as vu

SEED = 42
EVENTS = {"dog", "rooster", "sneezing", "crying_baby", "clock_tick"}
BACKGROUND = {"rain", "sea_waves", "crackling_fire", "helicopter", "chainsaw"}
NOISES = ["rain", "sea_waves", "wind", "engine", "vacuum_cleaner", "washing_machine",
          "crackling_fire", "keyboard_typing", "footsteps", "clapping"]
TRAIN_SPK = ["george", "jackson", "lucas", "nicolas"]
TEST_SPK = ["theo", "yweweler"]
SNRS = [20, 10, 5, 0]


def read_esc(root):
    with open(os.path.join(root, "meta", "esc50.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["x"] = vu.load_audio(os.path.join(root, "audio", r["filename"]))
    return rows


def part1(rows):
    ten = [r for r in rows if r["esc10"] == "True"]
    X = np.array([vu.clip_features(r["x"]) for r in ten])
    y10 = np.array([r["category"] for r in ten])
    yb = np.array([int(r["category"] in EVENTS) for r in ten])
    fold = np.array([int(r["fold"]) for r in ten])
    models = {"SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10)),
              "Neural network (MLP)": make_pipeline(StandardScaler(), MLPClassifier(
                  (64,), alpha=0.01, max_iter=800, random_state=SEED)),
              "Random forest": RandomForestClassifier(300, random_state=SEED)}
    out = {}
    for n, m in models.items():
        a10, ab, mf, pooled = [], [], [], np.zeros((10, 10), int)
        classes = sorted(set(y10.tolist()))
        for k in range(1, 6):
            tr, te = fold != k, fold == k
            p10 = m.fit(X[tr], y10[tr]).predict(X[te])
            a10.append(accuracy_score(y10[te], p10))
            mf.append(f1_score(y10[te], p10, average="macro"))
            pooled += confusion_matrix(y10[te], p10, labels=classes)
            ab.append(accuracy_score(yb[te], m.fit(X[tr], yb[tr]).predict(X[te])))
        out[n] = {"ten_class": [float(np.mean(a10)), float(np.std(a10, ddof=1))],
                  "macro_f1": [float(np.mean(mf)), float(np.std(mf, ddof=1))],
                  "classes": classes, "confusion": pooled.tolist(),
                  "event_vs_background": [float(np.mean(ab)), float(np.std(ab, ddof=1))]}
    return out


def loudest_second(x, rng):
    n = vu.SR
    if len(x) <= n:
        return np.pad(x, (0, n - len(x)))
    c = np.r_[0, np.cumsum(x ** 2)]
    return x[int(np.argmax(c[n:] - c[:-n])):][:n]


def part2(rows, fsdd, rng):
    def speech(spk):
        out = []
        for p in sorted(glob.glob(os.path.join(fsdd, "recordings", f"*_{spk}_*.wav")))[::2]:
            x = vu.load_audio(p)
            x = x / (np.abs(x).max() + 1e-9) * rng.uniform(0.1, 0.8)
            pad = max(vu.SR - len(x), 0)
            k = int(rng.integers(0, pad + 1))
            out.append(vu.reverb(np.pad(x, (k, pad - k))[:vu.SR] if pad else x[:vu.SR], rng))
        return out

    bg = {"train": [r["x"] for r in rows if r["category"] in NOISES[:7] and int(r["fold"]) <= 3],
          "test": [r["x"] for r in rows if r["category"] in NOISES[:7] and int(r["fold"]) >= 4]}

    def corrupt(x, snr, split):                  # same noise on both classes, so the class
        pool = bg[split]                         # difference cannot be the noise itself
        b = pool[int(rng.integers(len(pool)))]
        s = int(rng.integers(0, max(len(b) - len(x), 1)))
        b = b[s:s + len(x)]
        return x + vu.scale_to_snr(x, b, snr) * b

    train_ns = [r for r in rows if int(r["fold"]) <= 3]
    test_ns = [r for r in rows if int(r["fold"]) >= 4]
    Xtr, ytr = [], []
    for spk in TRAIN_SPK:
        for x in speech(spk):
            Xtr.append(vu.clip_features(corrupt(x, rng.choice(SNRS), "train"))); ytr.append(1)
    for r in train_ns:
        Xtr.append(vu.clip_features(corrupt(loudest_second(r["x"], rng), rng.choice(SNRS), "train"))); ytr.append(0)
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    models = {"SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, class_weight="balanced")),
              "Neural network (MLP)": make_pipeline(StandardScaler(), MLPClassifier(
                  (64,), alpha=0.01, max_iter=800, random_state=SEED))}
    for m in models.values():
        m.fit(Xtr, ytr)
    out = {"train_clips": len(ytr)}
    sp_test = [x for s in TEST_SPK for x in speech(s)]
    for snr in SNRS:
        Xs = np.array([vu.clip_features(corrupt(x, snr, "test")) for x in sp_test])
        Xn = np.array([vu.clip_features(corrupt(loudest_second(r["x"], rng), snr, "test")) for r in test_ns])
        X = np.vstack([Xs, Xn]); y = np.r_[np.ones(len(Xs)), np.zeros(len(Xn))]
        out[str(snr)] = {}
        for n, m in models.items():
            p = m.predict(X)
            bal = 0.5 * ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean())
            out[str(snr)][n] = {"balanced_accuracy": float(bal),
                                "speech_f1": float(f1_score(y, p)),
                                "speech_recall": float((p[y == 1] == 1).mean()),
                                "nonspeech_recall": float((p[y == 0] == 0).mean())}
            if snr == 10:
                bad = defaultdict(int)
                for r, q in zip(test_ns, p[len(Xs):]):
                    bad[r["category"]] += int(q == 1)
                out.setdefault("false_alarm_classes_10dB", {})[n] = sorted(
                    bad.items(), key=lambda kv: -kv[1])[:5]
    out["test_clips"] = {"speech": len(sp_test), "nonspeech": len(test_ns)}
    return out


def make_stream(spk_files, noises, snr, rng, seconds=20):
    n = seconds * vu.SR
    clean, truth = np.zeros(n), np.zeros(n, bool)
    t = int(rng.uniform(0.3, 1.0) * vu.SR)
    while True:
        u = vu.load_audio(spk_files[int(rng.integers(len(spk_files)))])
        if t + len(u) >= n:
            break
        u = u / (np.abs(u).max() + 1e-9)
        clean[t:t + len(u)] = u * rng.uniform(0.3, 1.0)
        env = np.convolve(u ** 2, np.ones(160) / 160, "same")
        truth[t:t + len(u)] = env > 10 ** (-3.5) * env.max()      # 35 dB below the utterance peak
        t += len(u) + int(rng.uniform(0.2, 1.5) * vu.SR)
    wet = vu.reverb(clean, rng)
    b = noises[int(rng.integers(len(noises)))]
    b = np.tile(b, n // len(b) + 1)[:n]
    x = wet + vu.scale_to_snr(wet, b, snr) * b
    labels = np.array([truth[i:i + vu.FRAME].mean() > 0.5
                       for i in range(0, n - vu.FRAME + 1, vu.HOP)])
    return x, labels


def part3(rows, fsdd, rng):
    noise = {"train": [r["x"] for r in rows if r["category"] in NOISES and int(r["fold"]) <= 3],
             "test": [r["x"] for r in rows if r["category"] in NOISES and int(r["fold"]) >= 4]}
    tr_files = [p for s in TRAIN_SPK for p in glob.glob(os.path.join(fsdd, "recordings", f"*_{s}_*.wav"))]
    te_files = [p for s in TEST_SPK for p in glob.glob(os.path.join(fsdd, "recordings", f"*_{s}_*.wav"))]
    X, y = [], []
    for i in range(48):
        x, lab = make_stream(sorted(tr_files), noise["train"], SNRS[i % 4], rng)
        keep = rng.choice(len(lab), 800, replace=False)
        X.append(vu.frame_features(x)[keep]); y.append(lab[keep])
    X, y = np.vstack(X), np.concatenate(y)
    print("VAD training frames", X.shape, "speech share", y.mean(), flush=True)
    learned = {"Logistic regression (frames)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
               "Neural network (frames)": make_pipeline(StandardScaler(), MLPClassifier(
                   (128, 64), alpha=1e-3, early_stopping=True, max_iter=300, random_state=SEED))}
    for m in learned.values():
        m.fit(X, y)
    # The two rule-based methods get a fair chance: their thresholds are tuned on
    # training streams, the same way the frame classifiers learn from training data.
    tune = [make_stream(sorted(tr_files), noise["train"], SNRS[i % 4], rng) for i in range(16)]
    grid = np.round(np.arange(0.15, 0.96, 0.05), 2)

    def tuned_alpha(rule):
        scores = []
        for alpha in grid:
            total = dict(tp=0, fp=0, fn=0, tn=0)
            for x, lab in tune:
                p = rule(x, alpha)
                L = min(len(p), len(lab))
                c = vu.frame_scores(lab[:L], p[:L])
                total = {k: total[k] + c[k] for k in total}
            scores.append(vu.from_counts(total)["f1"])
        return float(grid[int(np.argmax(scores))])

    alpha_e, alpha_h = tuned_alpha(vu.energy_vad), tuned_alpha(vu.entropy_vad)
    print("tuned thresholds", alpha_e, alpha_h, flush=True)
    counts = defaultdict(lambda: defaultdict(lambda: dict(tp=0, fp=0, fn=0, tn=0)))
    stream_f1 = defaultdict(lambda: defaultdict(list))
    speech_frames = total_frames = 0
    examples = {}
    for snr in SNRS:
        for i in range(16):
            x, lab = make_stream(sorted(te_files), noise["test"], snr, rng)
            F = vu.frame_features(x)
            speech_frames += int(lab.sum()); total_frames += len(lab)
            pred = {"Energy threshold": vu.energy_vad(x, alpha_e), "Spectral entropy": vu.entropy_vad(x, alpha_h)}
            for n, m in learned.items():
                pred[n] = vu.smooth(m.predict(F))
            for n, p in pred.items():
                L = min(len(p), len(lab))
                c = vu.frame_scores(lab[:L], p[:L])
                for k in c:
                    counts[n][snr][k] += c[k]
                stream_f1[n][snr].append(vu.from_counts(c)["f1"])
            if i == 0:
                examples[snr] = {"x": x, "truth": lab, **{k.replace(" ", "_"): v for k, v in pred.items()}}
    res = {n: {str(s): {**vu.from_counts(c), "stream_f1_std": float(np.std(stream_f1[n][s], ddof=1))}
               for s, c in d.items()} for n, d in counts.items()}
    return {"train_frames": len(y), "test_streams_per_snr": 16, "speech_share_train": float(y.mean()),
            "speech_share_test": speech_frames / total_frames,
            "tuned_alpha": {"energy": alpha_e, "entropy": alpha_h}, "scores": res}, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc", required=True)
    ap.add_argument("--fsdd", required=True)
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = read_esc(a.esc)
    print("loaded ESC-50", flush=True)
    res = {"part1": part1(rows)}
    print(res["part1"], flush=True)
    res["part2"] = part2(rows, a.fsdd, rng)
    print(res["part2"], flush=True)
    res["part3"], examples = part3(rows, a.fsdd, rng)
    print(json.dumps(res["part3"], indent=1), flush=True)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.savez(os.path.join(a.out, "vad_example.npz"), **examples[5])   # the report figure
    np.savez(os.path.join(a.out, "vad_examples_all.npz"),
             **{f"{s}/{k}": v for s, ex in examples.items() for k, v in ex.items()})


if __name__ == "__main__":
    main()
