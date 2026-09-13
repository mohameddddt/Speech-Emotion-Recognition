# Speech Emotion Recognition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohameddddt/Speech-Emotion-Recognition/blob/main/colab_notebook.ipynb)

Classifies the emotion in a short spoken sentence as **angry, disgust, fear, happy, neutral or sad**, using hand-crafted acoustic features and a classical ML ensemble trained on [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D).

## Results

Evaluated **speaker-independently**: 19 of the 91 actors are held out, so the test voices are never seen in training.

| model | accuracy | macro F1 |
|---|---|---|
| Random forest | 0.469 | 0.449 |
| MLP | 0.515 | 0.516 |
| SVM (RBF) | 0.516 | 0.511 |
| Logistic regression | 0.522 | 0.519 |
| XGBoost | 0.523 | 0.517 |
| **Soft-voting ensemble (LR + SVM + MLP, tuned)** | **0.532** | **0.531** |

Chance is 16.7%. The CREMA-D authors report that human raters identified the intended emotion from audio alone ~41% of the time.
Anger is recognised best (68% recall) and happiness worst (48%). Most confusions are between emotions with similar arousal.

## Project layout

```
data/AudioWAV/          7,442 CREMA-D clips (actor_sentence_emotion_intensity.wav)
ser/features.py         audio loading + 422-dim feature vector (MFCC/Δ/ΔΔ, log-mel, chroma,
                        spectral contrast/shape, RMS, ZCR, YIN pitch, duration)
ser/data.py             feature caching and actor-grouped train/test split
ser/model.py            ensemble definition, save/load, single-file prediction
train.py                train, evaluate on held-out actors, save artifacts/model.joblib
predict.py              classify WAV files with the saved model
project_notebook.ipynb  full walkthrough: EDA, features, model comparison, tuning, error analysis
colab_notebook.ipynb    same walkthrough, self-contained for Google Colab (+ upload your own audio)
```

## Run in Google Colab

Click the badge above and choose *Runtime → Run all*. The first cell clones this repo (including the audio) and installs the dependencies.
A full run takes ~25–30 minutes on the free CPU runtime; no GPU is needed.
The last cells let you upload your own recordings for prediction and download the trained model.

## Usage

```bash
pip install -r requirements.txt

python train.py            # ~2 min feature extraction on first run (cached), then ~1 min training
python train.py --full     # same report, but the saved model is refit on all 91 actors

python predict.py data/AudioWAV/1001_DFA_ANG_XX.wav
# data/AudioWAV/1001_DFA_ANG_XX.wav: angry  (angry 0.73, happy 0.25, fear 0.01, ...)
```

`predict.py` accepts any WAV/audio file librosa can read; audio is resampled to 16 kHz mono.
Generated files (feature cache, model, metrics) go to `artifacts/`, which is git-ignored.

## Possible next steps

- Fine-tune a self-supervised speech encoder (wav2vec 2.0 / HuBERT / WavLM). These usually beat hand-crafted features by a wide margin but need a GPU.
- Per-speaker feature normalisation when several utterances of the same speaker are available.
- Data augmentation (noise, pitch shift, speed perturbation) for robustness to real-world recordings.
