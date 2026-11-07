# Mini Task 9: Background Classification and VAD on VOiCES

Data: the VOiCES 90-degree subset (github.com/Lab41/VOiCES-subset): 908 files, 128 sentences,
104 speakers, 2 rooms, 4 microphone positions, backgrounds none / babble / music / television.
Every split is grouped by sentence. Events are tested separately on ESC-10.

- Report: [`report/MiniTask09_VOiCES_Report.pdf`](report/MiniTask09_VOiCES_Report.pdf)
- Interactive report with audio: `report/MiniTask09_VOiCES_Report.html` (download and open in Chrome)

## Background classification (4 classes, 5 folds, chance 25%)

| Classifier | Accuracy (%) | Macro F1 (%) |
|---|---|---|
| SVM (RBF) | 73.9 | 73.3 |
| Neural network (MLP) | 72.0 | 71.7 |
| Logistic regression | 72.6 | 72.0 |
| Random forest | 68.5 | 66.6 |
| k-NN | 60.3 | 57.7 |
| Gaussian naive Bayes | 62.0 | 60.1 |

## Frame-level VAD (about 80% of frames are speech)

| Method | F1 (%) | Miss rate (%) | False alarm rate (%) |
|---|---|---|---|
| Energy threshold | 95.6 | 4.1 | 18.7 |
| Spectral entropy | 88.3 | 9.0 | 59.9 |
| Logistic regression (frames) | 96.7 | 2.1 | 18.4 |
| Neural network (frames) | 97.0 | 1.9 | 16.6 |

VOiCES has no frame labels in this subset. The reference is built from the no-distractor
recording of each sentence and room, aligned to the other recordings by cross-correlation
(see Section 4 of the report).

## Run

    cd src
    # extract the eight VOiCES_90_*.tar archives so that all .wav files sit in one folder
    python run_voices_subset.py   --audio <wav folder> --out ../results
    python make_voices_figures.py --audio <wav folder>
    python run_task2.py --esc <ESC-50 folder> --fsdd <free-spoken-digit-dataset folder>   # ESC-10 part
    python make_figures.py                                                             # ESC-10 tables
    python vad_file.py --input <any wav> --out result                                  # VAD on one file

`run_task2.py` also contains my earlier simulated rehearsal (FSDD digits in a simulated room),
which is not part of the final report.

## Audio samples

`audio_samples/` has one sentence (sp3645, room rm1) with the four backgrounds (none, babble,
music, television), the music test file used in Figure 3 of the report, and two files made
from it at 8 kHz: the parts the neural network VAD keeps as speech and the parts it removes.
