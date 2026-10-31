# Speech Emotion Recognition

**Website and live demo: [emotion-speach-recognition.netlify.app](https://emotion-speach-recognition.netlify.app)**

| notebook | open |
|---|---|
| Pretrained speech embeddings (best model, GPU) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohameddddt/Speech-Emotion-Recognition/blob/main/notebooks/embeddings_colab.ipynb) |
| Hand-crafted features walkthrough (CPU) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohameddddt/Speech-Emotion-Recognition/blob/main/notebooks/colab_notebook.ipynb) |

Classifies the emotion in a short spoken sentence as **angry, disgust, fear, happy, neutral or sad**, trained on [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D).
Two approaches are compared: hand-crafted acoustic features with a classical ML ensemble, and embeddings from a frozen pretrained speech encoder with a linear classifier.

## Results

Evaluated **speaker-independently**: 19 of the 91 actors are held out, so the test voices are never seen in training. All models use the same split.

| model | accuracy | macro F1 |
|---|---|---|
| Random forest (hand-crafted) | 0.469 | 0.449 |
| MLP (hand-crafted) | 0.515 | 0.516 |
| SVM (hand-crafted) | 0.516 | 0.511 |
| Logistic regression (hand-crafted) | 0.522 | 0.519 |
| XGBoost (hand-crafted) | 0.523 | 0.517 |
| Soft-voting ensemble (hand-crafted) | 0.532 | 0.531 |
| **WavLM-base-plus embeddings (layers 5–9) + logistic regression** | **0.742** | **0.742** |

### Encoder comparison (5-fold actor-grouped CV, on Kaggle)

Nested cross-validation over all 91 actors: each fold tests on ~18 unseen actors, with layers and regularisation re-selected inside every fold.

| encoder (frozen) + logistic-regression probe | accuracy (mean ± std over folds) | macro F1 | layers used |
|---|---|---|---|
| **WavLM Large** | **0.779 ± 0.023** | **0.779** | 11 |
| WavLM Base+ | 0.743 ± 0.020 | 0.744 | 2–8 |
| HuBERT Base | 0.716 ± 0.025 | 0.716 | 1–12 |
| wav2vec 2.0 Base | 0.709 ± 0.026 | 0.709 | 1–12 |

With WavLM Large, per-class recall is neutral 89%, angry 87%, happy 81%, disgust 73%, sad 70% and fear 69%. The website serves all four probes, refit on all actors and temperature-calibrated.

Chance is 16.7%. The CREMA-D authors report that human raters identified the intended emotion from audio alone ~41% of the time.

With WavLM, per-class recall is angry 89%, neutral 88%, fear 75%, happy 69%, sad 66% and disgust 60%.
For the hand-crafted ensemble, anger was easiest (68%) and happiness hardest (48%).

## Project layout

```
data/AudioWAV/            7,442 CREMA-D clips (actor_sentence_emotion_intensity.wav)
ser/features.py           audio loading + 422-dim hand-crafted feature vector
ser/data.py               feature caching and actor-grouped train/test split
ser/model.py              hand-crafted ensemble: definition, save/load, prediction
ser/embeddings.py         frozen WavLM/HuBERT encoder, cached per-layer embeddings, linear probe
ser/probe.py              portable probe (.npz arrays, no pickle) + temperature calibration
ser/inference.py          Predictor: load once, classify raw arrays, sliding-window timeline
train.py                  train + evaluate the hand-crafted ensemble -> artifacts/model.joblib
train_embeddings.py       extract embeddings, select layers/C by grouped CV, evaluate
                          -> artifacts/embed_<encoder>.joblib
predict.py                classify audio files with either saved model
app/                      showcase website + live demo: server.py (FastAPI), web/ (page), models/, deploy to a Space
kaggle/                   Kaggle jobs: GPU embedding extraction, CPU evaluation + model export
tests/                    fast tests on synthetic audio (pytest)
notebooks/project_notebook.ipynb   hand-crafted walkthrough: EDA, features, model comparison, error analysis
notebooks/colab_notebook.ipynb     same walkthrough, self-contained for Colab
notebooks/embeddings_colab.ipynb   pretrained-embedding experiment for Colab (GPU), multiple encoders
docs/PLAN.md                       project assessment and roadmap
```

## Website

```bash
pip install -r app/requirements.txt
python app/server.py              # http://127.0.0.1:7860
```

A single page that presents the project and lets visitors try it:

- **Live demo**: record from the microphone, upload a file or pick a CREMA-D clip. Audio is decoded and resampled to 16 kHz in the browser and sent once as WAV. The result shows the emotion, calibrated probabilities, a waveform coloured by the emotion detected at each moment, and a timeline for recordings longer than ~4.5 s.
- **The story, with charts**: the model comparison, where emotion lives inside WavLM (per-layer probe accuracy), the confusion matrix with a model switch, the actor-grouped evaluation, limits and project status. Every chart has hover details and a table view. Light and dark themes.

Chart data comes from `app/results.json`, written by `app/update_models.py` (from the Kaggle evaluation, or `--legacy` for the original split). The models served are the portable probes in `app/models/`.

To publish it as a free Hugging Face Space (Docker): `hf auth login`, then `python app/deploy.py`.

## Training on Kaggle

All heavy compute runs on Kaggle, using the [`ejlok1/cremad`](https://www.kaggle.com/datasets/ejlok1/cremad) dataset, so nothing is cloned there.
`kaggle/push.py` embeds the working-tree `ser/` code in a single kernel script, so local changes don't have to be pushed to GitHub first.

```bash
python kaggle/push.py extract     # GPU: per-layer embeddings for wavlm-base-plus, hubert-base, wav2vec2-base, wavlm-large
python kaggle/push.py evaluate    # CPU: legacy split + nested 5-fold actor CV + all-actor probes for the website
python app/update_models.py       # download the evaluation output into app/models and app/results.json
```

## Run in Google Colab

Click a badge above and choose *Runtime → Run all*. The first cell clones this repo (including the audio) and installs what's missing.

- **`notebooks/embeddings_colab.ipynb`**: pick a **T4 GPU** runtime. Embeddings are cached to Google Drive, and extraction resumes after a disconnect. `ENCODERS_TO_RUN` selects the encoders (`wavlm-base-plus`, `hubert-base`, `wav2vec2-base`, `wavlm-large`).
- **`notebooks/colab_notebook.ipynb`**: CPU is fine; a full run takes ~25–30 minutes.

Both end with cells to upload your own recordings and download the trained model.

## Usage

```bash
pip install -r requirements.txt

# hand-crafted ensemble
python train.py            # ~2 min feature extraction on first run (cached), then ~1 min training
python train.py --full     # same report, but the saved model is refit on all 91 actors

# pretrained embeddings (additionally needs torch + transformers)
pip install torch transformers
python train_embeddings.py --encoder wavlm-base-plus    # GPU: minutes, CPU: ~1 h of extraction (cached)

python predict.py data/AudioWAV/1001_DFA_ANG_XX.wav
python predict.py --model artifacts/embed_wavlm-base-plus.joblib data/AudioWAV/1001_DFA_ANG_XX.wav
# data/AudioWAV/1001_DFA_ANG_XX.wav: angry  (angry 0.96, disgust 0.04, ...)
```

`predict.py` accepts any audio file librosa can read; audio is resampled to 16 kHz mono.
The embedding model file stores only the encoder name, the layers and the classifier (~56 KB); the encoder weights are downloaded from Hugging Face on first use.
Generated files (caches, models, metrics) go to `artifacts/`, which is git-ignored. `SER_DATA_DIR`, `SER_ARTIFACTS_DIR` and `SER_EMBED_DIR` override the default locations.

Tests: `pip install pytest && python -m pytest tests`.

## Possible next steps

See [docs/PLAN.md](docs/PLAN.md) for the full assessment and roadmap.

- Fine-tune the best encoder end to end (weighted layer sum + small head) with speed/noise augmentation.
- Per-speaker normalisation when several utterances of the same speaker are available.
