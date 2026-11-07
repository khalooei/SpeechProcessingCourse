# Mini Task 8: Yes/No Classification

Six classifiers (SVM, MLP neural network, logistic regression, random forest, k-NN,
Gaussian naive Bayes) on the Yes/No corpus (OpenSLR SLR1: 60 recordings, 8 kHz, one speaker,
eight Hebrew words per file, labels in the file name).

- Report: [`report/MiniTask08_YesNo_Report.pdf`](report/MiniTask08_YesNo_Report.pdf)
- Interactive report with audio: `report/MiniTask08_YesNo_Report.html` (download and open in Chrome)

## Main results

| Classifier | CV accuracy (%) | 0 dB, multi-condition training (%) |
|---|---|---|
| SVM (RBF) | 99.79 | 92.8 |
| Neural network (MLP) | 100.00 | 94.9 |
| Logistic regression | 100.00 | 93.8 |
| Random forest | 100.00 | 93.2 |
| k-NN | 99.75 | 74.2 |
| Gaussian naive Bayes | 99.79 | 55.7 |

Splits are always made per recording. One file (`0_1_0_1_0_0_0_0.wav`) is excluded because
only seven words are audible. Full tables, noise and filter bank comparisons are in the report.

## Run

    cd src
    python run_experiments.py --data ../data --out ../results   # downloads the data if needed
    python run_filterbank.py  --data ../data --out ../results
    python make_figures.py ../data ../results
    python make_tables.py ../results ../report/generated.tex

The audio is not uploaded; `run_experiments.py` downloads it into `data/`.

## Audio samples

`audio_samples/` has three recordings from the corpus so the data can be heard without
downloading it, for example `0_1_1_0_0_1_1_0.wav` (no, yes, yes, no, no, yes, yes, no).
All 60 recordings, with per-word playback and added noise, are inside the HTML report.
