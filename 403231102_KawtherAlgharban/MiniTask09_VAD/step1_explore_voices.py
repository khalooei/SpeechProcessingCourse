"""
Mini-Task 9 - Step 1: Download and explore the VOiCES subset
Digital Speech Processing - Dr. Mohammad Khalooei - AUT
Student: Kawther Algharban (403231102)

VOiCES = Voices Obscured in Complex Environmental Settings (Lab41 / SRI).
LibriSpeech sentences were replayed in real rooms and recorded by distant
microphones while a distractor sound was playing:
    none = no distractor, babb = babble, musi = music, tele = television

File name convention (distant speech):
  Lab41-SRI-VOiCES-<room>-<distractor>-sp<spk>-ch<chap>-sg<seg>-mc<mic>-<type>-<loc>-dg<deg>.wav

This script
  1. downloads the 90-degree subset (8 tar files, about 360 MB) from
     github.com/Lab41/VOiCES-subset, only if it is not there yet
  2. extracts the archives
  3. prints a summary: number of files, sample rate, durations,
     counts per distractor / room / microphone
  4. plays one example of each distractor type (macOS)
  5. saves waveform + spectrogram of one file per distractor type
"""

import re
import subprocess
import sys
import tarfile
import urllib.request
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "VOiCES-subset"
AUDIO_DIR = DATA_DIR / "audio"
FIG_DIR = BASE_DIR / "figures"
RAW = "https://github.com/Lab41/VOiCES-subset/raw/refs/heads/master/"
TARS = [f"VOiCES_90_{i}.tar" for i in range(1, 9)]
CSVS = ["Lab41-SRI-VOiCES_ref_SUBSET.csv",
        "Lab41-SRI-VOiCES-speaker-gender-dataset_SUBSET.csv"]

NAME_RE = re.compile(
    r"(?P<room>rm\d)-(?P<distractor>none|babb|musi|tele)"
    r"(?:-sp(?P<speaker>\d+)-ch(?P<chapter>\d+)-se?g(?P<segment>\d+))?"
    r"-mc(?P<mic>\d+)-(?P<mic_type>[a-z]+)-(?P<mic_loc>[a-z0-9]+)"
    r"(?:-dg(?P<degree>\d+))?"
)
DISTRACTOR_NAMES = {"none": "no distractor", "babb": "babble",
                    "musi": "music", "tele": "television"}


# ------------------------------------------------------------------
def fetch(name):
    """Download one file with a progress line. The file is first written as
    <name>.part and renamed only when complete, so an interrupted download
    is never mistaken for a finished one."""
    target = DATA_DIR / name
    part = DATA_DIR / (name + ".part")
    with urllib.request.urlopen(RAW + name, timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done, last = 0, -1
        while True:
            chunk = r.read(256 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            step = (10 * done // total) if total else int(done // 10e6)
            if step != last:
                last = step
                size = f" / {total / 1e6:.1f}" if total else ""
                print(f"   {name}: {done / 1e6:.1f}{size} MB", flush=True)
    part.rename(target)


def tar_ok(path):
    """True only if the whole archive can be read (catches truncated files)."""
    try:
        with tarfile.open(path) as t:
            t.getmembers()
        return True
    except Exception:
        return False


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in CSVS + TARS:
        target = DATA_DIR / name
        ok = target.exists() and target.stat().st_size > 0
        if ok and name.endswith(".tar") and not tar_ok(target):
            print(name, "is incomplete -> downloading again")
            ok = False
        if ok:
            continue
        print("Downloading", name, "...", flush=True)
        fetch(name)

    # The archives store absolute paths of the original author's computer
    # (e.g. /Users/.../file.wav), so we do NOT use extractall(). Every .wav
    # file is written directly into data/VOiCES-subset/audio/ instead.
    AUDIO_DIR.mkdir(exist_ok=True)
    marker = DATA_DIR / ".extracted"
    if not marker.exists():
        for name in TARS:
            print("Extracting", name, "...", flush=True)
            with tarfile.open(DATA_DIR / name) as tar:
                for m in tar:
                    base = Path(m.name).name
                    if m.isfile() and base.lower().endswith(".wav") and not base.startswith("._"):
                        with tar.extractfile(m) as src, open(AUDIO_DIR / base, "wb") as dst:
                            dst.write(src.read())
        marker.touch()


def load_wav(path):
    with wave.open(str(path), "rb") as f:
        sr, ch, width = f.getframerate(), f.getnchannels(), f.getsampwidth()
        raw = f.readframes(f.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:
        x = (x - 128) / 128
    else:
        x /= float(2 ** (8 * width - 1))
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def parse(path):
    m = NAME_RE.search(path.name)
    return m.groupdict() if m else None


# ------------------------------------------------------------------
def main():
    download()
    files = sorted(AUDIO_DIR.glob("*.wav"))
    print("\n========== Files ==========")
    print("Number of .wav files:", len(files))
    print("Example file names:")
    for p in files[:3]:
        print("   ", p.name)

    infos, unparsed = [], []
    for p in files:
        info = parse(p)
        if info is None:
            unparsed.append(p.name)
            continue
        with wave.open(str(p), "rb") as f:
            info["sr"] = f.getframerate()
            info["dur"] = f.getnframes() / f.getframerate()
        info["path"] = p
        infos.append(info)

    print("\n========== Summary ==========")
    print("Parsed file names :", len(infos), "| not parsed:", len(unparsed))
    if unparsed:
        print("   e.g.", unparsed[:3])
    print("Sample rate(s)    :", sorted({i["sr"] for i in infos}), "Hz")
    d = np.array([i["dur"] for i in infos])
    print("Duration (s)      : min %.2f | mean %.2f | max %.2f | total %.1f min"
          % (d.min(), d.mean(), d.max(), d.sum() / 60))
    for key in ["distractor", "room", "mic_type", "mic_loc", "degree"]:
        print(f"{key:<18}:", dict(sorted(Counter(i[key] for i in infos).items())))
    n_spk = len({i["speaker"] for i in infos if i["speaker"]})
    n_utt = len({(i["speaker"], i["chapter"], i["segment"]) for i in infos if i["speaker"]})
    print("Speakers          :", n_spk, "| different sentences:", n_utt)
    distractor_only = [i for i in infos if i["speaker"] is None]
    print("Distractor-only files (no speech):", len(distractor_only))

    # ---------- one example per distractor ----------
    examples = {}
    for i in infos:
        examples.setdefault(i["distractor"], i)
    order = [k for k in ["none", "babb", "musi", "tele"] if k in examples]

    if sys.platform == "darwin":
        for k in order:
            print(f"\nPlaying example: {DISTRACTOR_NAMES[k]}  ({examples[k]['path'].name})")
            subprocess.run(["afplay", "-t", "6", str(examples[k]["path"])])

    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(order), 2, figsize=(14, 3 * len(order)))
    axes = np.atleast_2d(axes)
    for row, k in zip(axes, order):
        x, sr = load_wav(examples[k]["path"])
        t = np.arange(len(x)) / sr
        row[0].plot(t, x, lw=0.4)
        row[0].set_title(f"{DISTRACTOR_NAMES[k]} — waveform", fontsize=10)
        row[0].set_xlabel("Time (s)")
        row[1].specgram(x, NFFT=400, Fs=sr, noverlap=240, cmap="magma")
        row[1].set_title(f"{DISTRACTOR_NAMES[k]} — spectrogram (25 ms / 10 ms)", fontsize=10)
        row[1].set_xlabel("Time (s)")
        row[1].set_ylabel("Hz")
    plt.tight_layout()
    out = FIG_DIR / "step1_voices_examples.png"
    plt.savefig(out, dpi=140)
    print("\nFigure saved to:", out)
    plt.show()


if __name__ == "__main__":
    main()
