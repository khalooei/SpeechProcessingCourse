# YESNO Spoken-Word Classification

**Student:** Baraa Atmeesh  
**Student ID:** 403131113  
**Course:** Digital Speech Processing  
**Instructor:** Dr. Khaloei  
**University:** Amirkabir University of Technology

## Project contents

- `Baraa_YesNo_Classification.ipynb`: complete notebook with explanations, executable Python cells and saved outputs.
- `Baraa_YesNo_Report.pdf`: final illustrated report.
- `run_experiment.py`: standalone Python implementation.
- `src/audio_pipeline.py`: word segmentation and spectral feature extraction.
- `data/waves_yesno/`: original YESNO audio recordings.
- `results/`: model results, test predictions, feature comparisons, run metadata and excluded-recording log.
- `figures/`: waveform, spectrogram, confusion matrices, comparison plots and per-recording segmentation reviews.
- `requirements.txt`: Python dependencies.

## Google Colab

1. Open Google Colab and upload `Baraa_YesNo_Classification.ipynb`.
2. Choose **Runtime > Run all** to reproduce the experiment. The notebook accesses the dataset with `torchaudio.datasets.YESNO` and downloads it when needed. The first download requires an Internet connection.
3. Open the `results` and `figures` folders in the Colab Files sidebar to find the regenerated outputs.

The notebook is self-contained: it does not import `src/audio_pipeline.py` or `run_experiment.py`.

## Windows / local Python

With Python 3.10 or newer, open a terminal in this project folder and run:

```bash
python -m pip install -r requirements.txt
python run_experiment.py
```

For recent torchaudio releases, install `torchcodec` if audio decoding reports that it is missing:

```bash
python -m pip install torchcodec
```

The audio recordings are already included in this package. `torchaudio.datasets.YESNO` will read them without downloading again.

## Evaluation

I evaluated five classifiers using a recording-grouped train/test split and grouped cross-validation within training data. The run excluded one recording with an unresolved mismatch between detected word bursts and its eight labels. The report and `results/run_metadata.json` give the resulting sample counts and outcomes. The saved segmentation review plots allow the boundaries to be checked again.
