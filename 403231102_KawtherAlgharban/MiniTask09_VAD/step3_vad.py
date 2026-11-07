"""
Mini-Task 9 - Step 3: Voice Activity Detection (VAD) on VOiCES
Digital Speech Processing - Dr. Mohammad Khalooei - AUT
Student: Kawther Algharban (403231102)

Question: in every 10 ms frame, is there speech or not - and how much do
background sounds (babble, music, TV) disturb different VAD methods?

Reference ("ground truth")
  VOiCES has no frame-level speech labels. Every sentence, however, was
  recorded in the SAME room with the SAME microphone in all four conditions.
  So we
    1. run Silero VAD on the CLEAN recording (distractor = none),
    2. align each noisy recording of the same sentence to a clean one (same
     room and microphone if available, otherwise the closest match) by
       cross-correlating their energy envelopes (search +-1 s),
    3. copy the clean speech/non-speech labels to the noisy recording.
  Recordings whose envelopes do not correlate well (r < 0.5) are skipped.
  The learned VAD is trained on a BALANCED sample (equal speech / non-speech),
  because about 85% of all frames are speech.

Compared VAD methods
  1. Energy VAD   : threshold on short-time log energy + 210 ms smoothing
  2. Learned VAD  : Neural network (MLP) on 19 frame features + 110 ms context,
                    trained on the reference labels of other sentences
                    (3-fold GroupKFold by sentence)
  3. Silero VAD   : pre-trained neural VAD (github.com/snakers4/silero-vad)
                    run directly on the noisy recording

Metrics per distractor type: accuracy, F1 (speech), false-alarm rate
(non-speech marked as speech) and miss rate (speech marked as non-speech).

Needs:  pip install silero-vad      (installs torch if missing)
"""

import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold

from audio_features import (list_files, load_wav, frame_features, CLASSES, CLASS_NAMES,
                            SR, FRAME, HOP, FIG_DIR, RES_DIR, FEAT_DIR)

try:
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps
except ImportError:
    raise SystemExit("Silero VAD is not installed. In the PyCharm Terminal run:\n"
                     "    pip install silero-vad\nthen run this script again.")

SEED = 42
MAX_LAG = 100            # +-1 s alignment search
MIN_CORR = 0.5
SMOOTH = 21              # 210 ms majority smoothing
CONTEXT = 11             # 110 ms context for the learned VAD
MAX_TRAIN_FRAMES = 150_000
METHODS = ["Energy VAD", "Learned VAD (MLP)", "Silero VAD"]


# ------------------------------------------------------------------ helpers
def majority_smooth(b, width=SMOOTH):
    k = np.ones(width) / width
    return np.convolve(b.astype(float), k, mode="same") > 0.5


def moving_mean(F, width=CONTEXT):
    k = np.ones(width) / width
    return np.stack([np.convolve(F[:, j], k, mode="same") for j in range(F.shape[1])], axis=1)


def silero_labels(model, x, n_frames):
    ts = get_speech_timestamps(torch.from_numpy(x.copy()), model, sampling_rate=SR)
    centers = np.arange(n_frames) * HOP + FRAME // 2
    lab = np.zeros(n_frames, dtype=bool)
    for seg in ts:
        lab |= (centers >= seg["start"]) & (centers < seg["end"])
    return lab


def energy_vad(log_e, alpha=0.35):
    e = np.convolve(log_e, np.ones(5) / 5, mode="same")
    floor, peak = np.percentile(e, 10), np.percentile(e, 99)
    return majority_smooth(e > floor + alpha * (peak - floor))


def best_lag(noisy_e, clean_e):
    """lag L such that noisy[t] ~ clean[t - L]; returns (L, correlation)."""
    a = (noisy_e - noisy_e.mean()) / (noisy_e.std() + 1e-9)
    b = (clean_e - clean_e.mean()) / (clean_e.std() + 1e-9)
    best = (0, -1.0)
    for L in range(-MAX_LAG, MAX_LAG + 1):
        lo = max(0, L)
        hi = min(len(a), len(b) + L)
        if hi - lo < 100:
            continue
        r = np.corrcoef(a[lo:hi], b[lo - L:hi - L])[0, 1]
        if r > best[1]:
            best = (L, r)
    return best


def metrics(ref, hyp):
    tp = np.sum(ref & hyp)
    tn = np.sum(~ref & ~hyp)
    fp = np.sum(~ref & hyp)
    fn = np.sum(ref & ~hyp)
    return dict(acc=(tp + tn) / len(ref), f1=2 * tp / max(2 * tp + fp + fn, 1),
                far=fp / max(fp + tn, 1), miss=fn / max(fn + tp, 1))


# ------------------------------------------------------------------ main
def main():
    files = list_files()
    by_sentence = defaultdict(list)
    for f in files:
        by_sentence[f["sentence"]].append(f)
    n_clean = sum(any(f["distractor"] == "none" for f in v) for v in by_sentence.values())
    print(f"Recordings: {len(files)} | sentences: {len(by_sentence)} | sentences with a clean version: {n_clean}")

    cache = FEAT_DIR / "step3_vad_data_v2.npz"
    model = load_silero_vad()

    if cache.exists():
        print("Loaded cached frame data:", cache.name)
        d = np.load(cache, allow_pickle=True)
        X, ref, sil, eng = d["X"], d["ref"], d["sil"], d["eng"]
        cond, sent, fid, info = d["cond"], d["sent"], d["fid"], list(d["info"])
    else:
        X, ref, sil, eng, cond, sent, fid, info = [], [], [], [], [], [], [], []
        clean_cache = {}                                   # path -> (log energy, Silero labels)

        def clean_ref(f):
            if f["path"] not in clean_cache:
                xc, _ = load_wav(f["path"])
                Fc = frame_features(xc)
                clean_cache[f["path"]] = (Fc[:, 13], silero_labels(model, xc, len(Fc)))
            return clean_cache[f["path"]]

        lags, corrs, skipped, t0 = [], [], 0, time.time()
        for i, f in enumerate(files):
            cleans = [c for c in by_sentence[f["sentence"]] if c["distractor"] == "none"]
            # prefer: same room + same microphone, then same room, then any room
            cleans.sort(key=lambda c: (c["room"] != f["room"], c["mic_loc"] != f["mic_loc"]))
            x, _ = load_wav(f["path"])
            F = frame_features(x)
            found = None
            for c in cleans[:3]:
                ce, cref = clean_ref(c)
                L, r = (0, 1.0) if c["path"] == f["path"] else best_lag(F[:, 13], ce)
                if r >= MIN_CORR:
                    found = (L, r, cref)
                    break
            if found is None:
                skipped += 1
                continue
            L, r, ref_clean = found
            lags.append(L)
            corrs.append(r)
            src = np.arange(len(F)) - L
            valid = (src >= 0) & (src < len(ref_clean))
            lab = np.zeros(len(F), dtype=bool)
            lab[valid] = ref_clean[src[valid]]
            n = valid.sum()
            X.append(np.hstack([F, moving_mean(F)])[valid].astype(np.float32))
            ref.append(lab[valid])
            sil.append(silero_labels(model, x, len(F))[valid])
            eng.append(energy_vad(F[:, 13])[valid])
            cond += [f["distractor"]] * n
            sent += [f["sentence"]] * n
            fid += [len(info)] * n
            info.append(f"{f['path'].name}|{L}|{r:.2f}")
            if (i + 1) % 100 == 0 or i + 1 == len(files):
                print(f"   processed {i + 1}/{len(files)} recordings ({time.time() - t0:.0f} s)", flush=True)
        print(f"Aligned recordings: {len(info)} | skipped (no clean match with r >= {MIN_CORR}): {skipped}")
        print(f"Alignment: median |lag| = {np.median(np.abs(lags)) * 10:.0f} ms | "
              f"median correlation = {np.median(corrs):.2f}")
        X, ref, sil, eng = np.vstack(X), np.concatenate(ref), np.concatenate(sil), np.concatenate(eng)
        cond, sent, fid = np.array(cond), np.array(sent), np.array(fid)
        FEAT_DIR.mkdir(exist_ok=True)
        np.savez(cache, X=X, ref=ref, sil=sil, eng=eng, cond=cond, sent=sent, fid=fid,
                 info=np.array(info, dtype=object))
    for c in CLASSES:
        print(f"   {CLASS_NAMES[c]:<14}: {len(set(fid[cond == c]))} recordings")

    print(f"Frames: {len(ref):,} | speech in reference: {ref.mean() * 100:.1f}%")

    # ---------------- learned VAD: 3-fold grouped by sentence ----------------
    learned = np.zeros_like(ref)
    rng = np.random.default_rng(SEED)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=3).split(X, ref, sent)):
        # balanced sample: as many non-speech as speech frames (speech is ~85% of frames)
        pos, neg = tr[ref[tr]], tr[~ref[tr]]
        k_each = min(len(pos), len(neg), MAX_TRAIN_FRAMES // 2)
        if k_each > 0:
            tr = np.concatenate([rng.choice(pos, k_each, replace=False),
                                 rng.choice(neg, k_each, replace=False)])
        elif len(tr) > MAX_TRAIN_FRAMES:
            tr = rng.choice(tr, MAX_TRAIN_FRAMES, replace=False)
        clf = make_pipeline(StandardScaler(),
                            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                                          early_stopping=True, random_state=SEED))
        t0 = time.time()
        clf.fit(X[tr], ref[tr])
        raw = clf.predict(X[te]).astype(bool)
        for f in np.unique(fid[te]):                 # smooth inside each recording
            m = te[fid[te] == f]
            learned[m] = majority_smooth(raw[fid[te] == f])
        print(f"   learned VAD fold {k + 1}/3 trained on {len(tr):,} balanced frames ({time.time() - t0:.0f} s)")

    hyps = {"Energy VAD": eng, "Learned VAD (MLP)": learned, "Silero VAD": sil}

    # ---------------- results table ----------------
    rows = []
    print(f"\n{'Condition':<16}{'Method':<20}{'Accuracy':>9}{'F1':>7}{'False alarm':>13}{'Miss':>7}")
    print("-" * 72)
    present = [c for c in CLASSES if np.any(cond == c)]
    for c in present:
        m = cond == c
        for name in METHODS:
            r = metrics(ref[m], hyps[name][m].astype(bool))
            rows.append((c, name, r))
            note = "  (reference itself)" if (c == "none" and name == "Silero VAD") else ""
            print(f"{CLASS_NAMES[c]:<16}{name:<20}{r['acc'] * 100:>8.1f}%{r['f1']:>7.3f}"
                  f"{r['far'] * 100:>12.1f}%{r['miss'] * 100:>6.1f}%{note}")
    print("\nAll noisy conditions together (babble + music + TV):")
    noisy = cond != "none"
    for name in METHODS:
        r = metrics(ref[noisy], hyps[name][noisy].astype(bool))
        print(f"   {name:<20} accuracy {r['acc'] * 100:5.1f}% | F1 {r['f1']:.3f} | "
              f"false alarm {r['far'] * 100:4.1f}% | miss {r['miss'] * 100:4.1f}%")

    RES_DIR.mkdir(exist_ok=True)
    with open(RES_DIR / "step3_vad_results.csv", "w") as f:
        f.write("condition,method,accuracy,f1,false_alarm_rate,miss_rate\n")
        for c, name, r in rows:
            f.write(f"{c},{name},{r['acc']:.4f},{r['f1']:.4f},{r['far']:.4f},{r['miss']:.4f}\n")

    FIG_DIR.mkdir(exist_ok=True)
    # ---------------- Figure 1: F1 / false alarm / miss by condition ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    pos = np.arange(len(present))
    for ax, (key, title) in zip(axes, [("f1", "F1 (speech) - higher is better"),
                                       ("far", "False-alarm rate % - lower is better"),
                                       ("miss", "Miss rate % - lower is better")]):
        for k, name in enumerate(METHODS):
            vals = [next(r for c, n, r in rows if c == cl and n == name)[key] for cl in present]
            vals = [v if key == "f1" else v * 100 for v in vals]
            ax.bar(pos + (k - 1) * 0.27, vals, 0.27, label=name)
        ax.set_xticks(pos)
        ax.set_xticklabels([CLASS_NAMES[c] for c in present], rotation=10)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("VAD performance by background condition (reference: Silero on the aligned clean recording)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step3_vad_comparison.png", dpi=150)

    # ---------------- Figure 2: one sentence in all four conditions ----------------
    files_by_name = {f["path"].name: f for f in list_files()}
    per_sentence = defaultdict(dict)                 # sentence -> {distractor: info index}
    for i, s_ in enumerate(info):
        f = files_by_name[s_.split("|")[0]]
        per_sentence[f["sentence"]].setdefault(f["distractor"], i)
    best_sentence = max(per_sentence, key=lambda k: len(per_sentence[k]))
    shown = [per_sentence[best_sentence][c] for c in CLASSES if c in per_sentence[best_sentence]]
    fig, axes = plt.subplots(len(shown), 1, figsize=(14, 2.6 * len(shown)), sharex=True)
    axes = np.atleast_1d(axes)
    colors = {"Energy VAD": "tab:red", "Learned VAD (MLP)": "tab:green", "Silero VAD": "tab:purple"}
    for ax, i in zip(axes, shown):
        name = info[i].split("|")[0]
        m = fid == i
        x, _ = load_wav(files_by_name[name]["path"])
        t = np.arange(len(x)) / SR
        ax.plot(t, x / (np.abs(x).max() + 1e-9), color="gray", lw=0.4)
        tf = (np.arange(m.sum()) * HOP + FRAME / 2) / SR
        ax.fill_between(tf, -1, 1, where=ref[m], color="gold", alpha=0.3, step="mid",
                        label="reference speech")
        for k, mth in enumerate(METHODS):
            ax.fill_between(tf, -1.25 - 0.22 * k, -1.07 - 0.22 * k, where=hyps[mth][m].astype(bool),
                            color=colors[mth], step="mid", label=mth)
        ax.set_ylim(-1.95, 1.05)
        ax.set_yticks([])
        ax.set_title(f"{CLASS_NAMES[files_by_name[name]['distractor']]}  —  {name}", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=7, ncol=4)
    axes[-1].set_xlabel("Time (s)  — bars under each waveform: Energy (red), Learned (green), Silero (purple)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step3_vad_example.png", dpi=150)
    print("Figures saved to:", FIG_DIR)
    plt.show()


if __name__ == "__main__":
    main()
