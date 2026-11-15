# Mini Task 09 — VOiCES Background Classification and Voice Activity Detection

Baraa Atmeesh  
Student ID: 403131113  
Amirkabir University of Technology  
Digital Speech Processing — Dr. Khaloei

## Project overview

I used the VOiCES 90-degree subset to study two related speech-processing tasks: classification of the recording background (none, babble, music, or television) and frame-level voice activity detection (VAD). I compared several classification models and examined how different acoustic conditions affect detection performance.

## Files

- `Baraa_MT9_VOiCES.ipynb` — Google Colab notebook with code, explanations, saved results, and figures.
- `Baraa_MT9_Code.py` — standalone Python version of the experiment.
- `Baraa_MT9_Report.pdf` — report for reading and submission.
- `figures/` — plots from the experiment.
- `results/` — summary tables and experiment metadata.
- `requirements.txt` — Python dependencies.

## How to run

1. Open Google Colab and upload `Baraa_MT9_VOiCES.ipynb`.
2. Select **Runtime → Run all**.
3. Allow the notebook to download the VOiCES subset and install its required packages.
4. Review the generated figures and result tables in `/content/Baraa_MT9/`.

An Internet connection and sufficient storage are required for the dataset download. Results may vary slightly between runs depending on the software environment and model initialization.

## Evaluation note

For VAD evaluation, the reference labels are generated automatically by Silero VAD on the corresponding close-microphone audio. They are not human-annotated ground truth, so scores measure agreement with this automatic reference.
