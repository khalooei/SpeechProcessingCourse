"""
Mini-Task 8 - Step 1: Data exploration (YesNo dataset)
Digital Speech Processing - Dr. Mohammad Khalooei - AUT
Student: Kawther Algharban (403231102)

What this script does:
  1. Downloads the YesNo dataset (OpenSLR #1) into ./data if it is not there.
  2. Prints basic facts: number of files, sample rate, duration, labels.
  3. Plays one sample (macOS only).
  4. Saves a waveform plot and a spectrogram plot into ./figures.
"""

import subprocess
import sys
import tarfile
import urllib.request
import wave
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Paths (relative to this file, so it works on any computer)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WAV_DIR = DATA_DIR / "waves_yesno"
FIG_DIR = BASE_DIR / "figures"
URL = "https://www.openslr.org/resources/1/waves_yesno.tar.gz"


def download_dataset():
    """Download and extract the dataset only if it is missing."""
    if WAV_DIR.exists() and any(WAV_DIR.glob("*.wav")):
        print("Dataset already exists:", WAV_DIR)
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / "waves_yesno.tar.gz"
    print("Downloading YesNo dataset ...")
    urllib.request.urlretrieve(URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(DATA_DIR)
    archive.unlink()  # remove the archive, keep only the .wav files
    print("Done. Files saved to:", WAV_DIR)


def load_wav(path):
    """Read a 16-bit mono wav file -> (signal in [-1, 1], sample_rate)."""
    with wave.open(str(path), "rb") as f:
        sr = f.getframerate()
        frames = f.readframes(f.getnframes())
    signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return signal, sr


def labels_from_name(path):
    """'0_1_1_0_0_1_1_0.wav' -> [0, 1, 1, 0, 0, 1, 1, 0]   (1 = yes, 0 = no)"""
    return [int(x) for x in path.stem.split("_")]


def main():
    download_dataset()
    files = sorted(WAV_DIR.glob("*.wav"))

    # ---------------- 1. Basic facts ----------------
    print("\n========== Dataset summary ==========")
    print("Number of audio files :", len(files))

    durations, rates = [], set()
    n_yes = n_no = 0
    for p in files:
        sig, sr = load_wav(p)
        rates.add(sr)
        durations.append(len(sig) / sr)
        lab = labels_from_name(p)
        n_yes += sum(lab)
        n_no += len(lab) - sum(lab)

    print("Sample rate(s)        :", rates, "Hz")
    print("Duration (sec)        : min %.2f | mean %.2f | max %.2f"
          % (min(durations), np.mean(durations), max(durations)))
    print("Words per file        : 8")
    print("Total words           :", n_yes + n_no,
          "(yes = %d, no = %d)" % (n_yes, n_no))

    # ---------------- 2. One example ----------------
    example = files[0]
    signal, sr = load_wav(example)
    print("\n========== Example file ==========")
    print("File name   :", example.name)
    print("Labels      :", labels_from_name(example), "(1 = yes, 0 = no)")
    print("Samples     :", len(signal))
    print("Duration    : %.2f sec" % (len(signal) / sr))

    # Play the sound (macOS has the built-in 'afplay' command)
    if sys.platform == "darwin":
        print("\nPlaying the example ...")
        subprocess.run(["afplay", str(example)])

    # ---------------- 3. Plots ----------------
    FIG_DIR.mkdir(exist_ok=True)
    t = np.arange(len(signal)) / sr

    fig, ax = plt.subplots(2, 1, figsize=(12, 7))

    ax[0].plot(t, signal, linewidth=0.6)
    ax[0].set_title(f"Waveform: {example.name}   labels = {labels_from_name(example)}")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlim(0, t[-1])

    # Spectrogram: 25 ms window (200 samples at 8 kHz), 10 ms hop
    ax[1].specgram(signal, NFFT=200, Fs=sr, noverlap=120, cmap="magma")
    ax[1].set_title("Spectrogram (STFT, 25 ms window, 10 ms hop)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Frequency (Hz)")

    plt.tight_layout()
    out = FIG_DIR / "step1_waveform_spectrogram.png"
    plt.savefig(out, dpi=150)
    print("\nFigure saved to:", out)
    plt.show()


if __name__ == "__main__":
    main()
