# Speech Emotion Recognition

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
train.py                  train + evaluate the hand-crafted ensemble -> artifacts/model.joblib
train_embeddings.py       extract embeddings, select layers/C by grouped CV, evaluate
                          -> artifacts/embed_<encoder>.joblib
predict.py                classify audio files with either saved model
project_notebook.ipynb    hand-crafted walkthrough: EDA, features, model comparison, error analysis
colab_notebook.ipynb      same walkthrough, self-contained for Colab
embeddings_colab.ipynb    pretrained-embedding experiment for Colab (GPU), multiple encoders
```

## Run in Google Colab

Click a badge above and choose *Runtime → Run all*. The first cell clones this repo (including the audio) and installs what's missing.

- **`embeddings_colab.ipynb`**: pick a **T4 GPU** runtime. Embeddings are cached to Google Drive, and extraction resumes after a disconnect. `ENCODERS_TO_RUN` selects the encoders (`wavlm-base-plus`, `hubert-base`, `wav2vec2-base`, `wavlm-large`).
- **`colab_notebook.ipynb`**: CPU is fine; a full run takes ~25–30 minutes.

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
Generated files (caches, models, metrics) go to `artifacts/`, which is git-ignored.

## Possible next steps

- Compare `hubert-base` and `wavlm-large` with `embeddings_colab.ipynb`.
- Fine-tune the best encoder end to end (weighted layer sum + small head) with speed/noise augmentation.
- Per-speaker normalisation when several utterances of the same speaker are available.
