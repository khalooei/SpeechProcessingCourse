# Mini-Task 8 — Yes/No Audio Classification

**Course:** Digital Speech Processing  
**Instructor:** Dr. Khalooei  
**Student:** Anas Raad Radih Alaasam  
**Student ID:** 404131083  
**Email:** anas.r.r.alaasam@gmail.com

## Objective

I used TorchAudio's YESNO dataset to train and compare four classifiers: Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Random Forest, and a Multi-Layer Perceptron (MLP) neural network.

## Dataset and method

- Source: TorchAudio `torchaudio.datasets.YESNO` (originally OpenSLR SLR1).
- Dataset: 60 recordings, 8 binary labels per recording, 8 kHz mono audio; `0 = No`, `1 = Yes`.
- I applied short-time-energy segmentation and expanded the detected intervals with a 50 ms margin. With the revised thresholds, 59 recordings contained the expected eight detected speech regions.
- I excluded recording **21**, because the segmentation algorithm detected seven regions for eight labels and I could not establish reliable alignment.
- I used **472 candidate word segments** from 59 recordings: 223 No and 249 Yes.
- Each segment was represented by **32 acoustic features**: 13 MFCC means, 13 MFCC standard deviations, and the mean and standard deviation of each of ZCR, RMS energy, and spectral centroid.
- I used `GroupShuffleSplit` to split the data by original recording: 376 training segments from 47 recordings and 96 test segments from 12 recordings, with no source-recording overlap.
- SVM, KNN, and MLP use training-fitted standardization in scikit-learn pipelines. The same test partition was used for all classifiers.

## Results

The following values are taken from the **saved outputs of the submitted Colab notebook**, not from a new execution.

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1 (Yes) | Fit time (s) |
|---|---:|---:|---:|---:|---:|
| SVM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.017912 |
| KNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.005055 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.413030 |
| MLP Neural Network | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.163639 |

For each model, the test-set confusion matrix (rows: actual No/Yes, columns: predicted No/Yes) was `[[43, 0], [0, 53]]`. Five-fold `GroupKFold` cross-validation applied **only to the training split** returned accuracy and F1 of 1.0000 in all folds for all four models. Results are also available as CSV files in `results/`.

## Figures

The `figures/` folder contains image outputs extracted from the submitted notebook, including waveform/spectrogram and confusion-matrix plots. The executed notebook retains all inline outputs.

## How to run

1. Open `MT8_YesNo_Anas_404131083.ipynb` in Google Colab.
2. Use a Python runtime with compatible PyTorch and TorchAudio packages. The original notebook recorded Python 3.13.18, PyTorch 2.14.1+cpu, and TorchAudio 2.11.0+cpu.
3. Run the cells **top to bottom**. The notebook installs basic dependencies, downloads the dataset through TorchAudio, creates `/content/YESNO_Dataset`, and performs segmentation, feature extraction, training, and evaluation.
4. No local audio dataset or model weight files are bundled in this submission. Internet access is required for the initial dataset download.

`requirements.txt` lists the packages used; its versions are unpinned because the exact complete environment was not recorded. Compatibility may require adjusting PyTorch/TorchAudio versions together.

## Interpretation and limitations

All four models achieved 100% accuracy on the held-out recordings in this specific experiment; **this is not evidence of universal accuracy**. The dataset contains one speaker, so speaker-independent generalization was not tested. Moreover, automatic segmentation matching eight detected regions to eight labels does not by itself establish correct word boundaries or label alignment for every segment. The result should be interpreted with those limitations. Grouped evaluation prevents sharing an original recording between training and test sets, but does not eliminate every possible recording artifact or shortcut.

## References

- TorchAudio documentation: https://docs.pytorch.org/audio/stable/index.html
- OpenSLR SLR1 (YESNO): https://www.openslr.org/1/
- Course repository: https://github.com/khalooei/SpeechProcessingCourse

## Submission note

This directory is prepared for copying into a student-owned **fork** of the course repository. No remote GitHub upload has been performed by this packaging step. The student directory name is a proposed name and should be checked against the instructor's exact naming rules.
