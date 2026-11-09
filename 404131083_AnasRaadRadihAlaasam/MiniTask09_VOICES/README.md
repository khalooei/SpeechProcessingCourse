# MT9 - Background Sound Classification and Voice Activity Detection

**Amirkabir University of Technology**  
**Department of Computer Engineering**  
**Course:** Digital Speech Processing  
**Instructor:** Dr. Mohammad Khalooei  
**Student:** Anas Raad Radih Alaasam  
**Student ID:** 404131083  
**Email:** anas.r.r.alaasam@gmail.com  
**Date:** November 2026

## Overview

In this mini-task, I used VOiCES-subset to study background sound classification and voice activity detection (VAD). I classified the background condition of whole recordings as no distractor (`none`), babble (`babb`), music (`musi`), or television (`tele`). I also compared classical energy-based VAD with pretrained Silero VAD to identify probable speech intervals.

For the background task, I evaluated recording-level categories rather than the start and end times of individual sound events. I kept that distinction explicit in my discussion.

## Files

I included the executed notebook, my report, the figures, and summary tables under the following names:

| File or folder | Contents |
|---|---|
| `MT9_VOICES_Anas_404131083.ipynb` | My Google Colab notebook with code, saved outputs, and discussion. |
| `MT9_VOICES_Report_Anas_404131083.docx` | My editable report. |
| `MT9_VOICES_Report_Anas_404131083.pdf` | My PDF report. |
| `README.md` | Project description, experiment summary, and reproduction steps. |
| `requirements.txt` | Python packages used by the notebook. |
| `figures/` | Eleven figures, including a four-panel comparison for readability. |
| `results/` | Five CSV tables summarizing the experimental results. |

I did not include the raw VOiCES audio files. My notebook downloads the publicly available subset from its repository and extracts the recordings inside Colab.

## Dataset and train/test split

I inspected eight audio archives and identified 909 WAV-named members, including one non-audio AppleDouble entry. I excluded that non-audio entry and retained **908 valid mono recordings at 16 kHz**. I found the following class distribution: 227 no-distractor, 228 babble, 231 music, and 222 television recordings.

I identified 104 speakers and used a speaker-grouped split. I trained on **666 recordings from 78 speakers** and tested on **242 recordings from 26 different speakers**. I checked that the two sets had no overlapping speaker IDs.

## Methods

I extracted 30 acoustic statistics per recording: the means and standard deviations of 13 MFCC coefficients, plus RMS energy mean/std and zero-crossing rate mean/std. I trained a `StandardScaler` + RBF `SVC` pipeline (`C=1`, `gamma="scale"`) using the whole-recording features.

I then selected the 25% lowest-RMS analysis frames in each recording and computed the same 30 statistics from those frames. I kept the same SVM configuration and the same speaker-disjoint train/test split. I also performed five-fold `GroupKFold` cross-validation on the training speakers only.

For VAD, I implemented a classical energy-based method and applied pretrained Silero VAD to the same recordings. I began with four examples, then selected 10 recordings per background condition (**40 recordings total**) for a wider comparison. I compared their frame labels and speech ratios; I did not treat agreement as ground-truth detection accuracy.

## Results

| Experiment | My result |
|---|---:|
| Whole-recording SVM accuracy | **69.42%** (168/242) |
| Low-energy SVM accuracy | **87.60%** (212/242) |
| Accuracy improvement | **18.18 percentage points** |
| Five-fold grouped CV: whole-recording mean accuracy | **70.71%** |
| Five-fold grouped CV: low-energy mean accuracy | **87.83%** |
| Mean Energy VAD / Silero VAD frame agreement | **81.70%** across 40 recordings |
| Exact paired McNemar test | **p ≈ 5.21 × 10⁻⁹** (exploratory) |

I found that the low-energy SVM produced 44 more correct predictions than the whole-recording SVM on the same 242 recordings. I also observed that Silero labeled a larger fraction of the analyzed frames as speech than my energy-based detector. Without independent speech annotations, I cannot conclude that either VAD method is more accurate.

## Google Colab reproduction

I tested the notebook from a fresh Google Colab runtime by taking these steps:

1. I opened `MT9_VOICES_Anas_404131083.ipynb` in Colab.
2. I selected **Runtime > Disconnect and delete runtime**, then connected to a new runtime.
3. I selected **Runtime > Run all** and allowed the audio files and Silero model to download.
4. I checked that all **25 code cells** completed without execution errors.
5. I inspected the resulting metrics and figures.

For a new run, I need an internet connection for the VOiCES-subset repository and any missing Silero model files. My experiments were run in Google Colab with Python 3.13.18. I did not pin every library version, so exact results could vary after upstream updates.

I saved experiment results under `/content/MT9_results/` and figures under `/content/MT9_figures/`. I included summary CSV tables and figure images in this folder. I also kept the complete experimental code and its execution outputs in my notebook.

## Limitations

I evaluated background categories at the recording level; I did not implement temporal onset/offset detection for individual background events. My low-energy frames were selected by RMS energy rather than verified speech pauses. I did not have independent frame-level VAD reference labels, so I reported detector agreement instead of VAD accuracy or F1-score. I also interpret the McNemar comparison cautiously because some recordings share speakers and I examined the held-out test results during exploratory feature comparison.

## References

1. C. Richey et al., “Voices Obscured in Complex Environmental Settings (VOiCES) corpus,” *Interspeech*, 2018. https://www.isca-archive.org/interspeech_2018/richey18_interspeech.pdf
2. Lab41, “VOiCES-subset,” https://github.com/Lab41/VOiCES-subset
3. Silero VAD, https://github.com/snakers4/silero-vad
4. M. Fabien, “Voice Activity Detection Application,” https://maelfabien.github.io/project/Speech_proj/#high-level-overview
