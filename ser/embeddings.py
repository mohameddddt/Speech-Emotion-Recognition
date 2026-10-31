"""Utterance embeddings from pretrained self-supervised speech models (WavLM, HuBERT, ...).

Each clip is passed through a frozen encoder; the hidden states of every layer are
mean-pooled over time and cached as a (n_clips, n_layers + 1, dim) float16 array, so the
classifier can later pick the best layer without re-running the encoder.
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ser.data import ARTIFACTS_DIR
from ser.features import SAMPLE_RATE, load_audio

ENCODERS = {
    "wavlm-base-plus": "microsoft/wavlm-base-plus",
    "hubert-base": "facebook/hubert-base-ls960",
    "wav2vec2-base": "facebook/wav2vec2-base",
    "wavlm-large": "microsoft/wavlm-large",
}
EMBED_DIR = Path(os.environ.get("SER_EMBED_DIR", ARTIFACTS_DIR / "embeddings"))
CHUNK = 500


def default_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class Encoder:
    """Frozen pretrained speech encoder returning per-layer mean-pooled embeddings."""

    def __init__(self, name, device=None):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        self.name, self.hf_id = name, ENCODERS.get(name, name)
        self.device = device or default_device()
        self.processor = AutoFeatureExtractor.from_pretrained(self.hf_id)
        self.model = AutoModel.from_pretrained(self.hf_id).to(self.device).eval()
        self._torch = torch

    def __call__(self, y):
        torch = self._torch
        inputs = self.processor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            out = self.model(inputs.input_values.to(self.device), output_hidden_states=True)
        hidden = torch.stack(out.hidden_states)[:, 0]  # (layers + 1, frames, dim)
        return hidden.float().mean(dim=1).cpu().numpy().astype(np.float16)

    def frames(self, y, layers, chunk_seconds=20):
        """Per-frame (20 ms) embeddings averaged over `layers`: (n_frames, dim) float32.

        Long audio is encoded in chunks so attention memory stays bounded.
        """
        torch, step, out = self._torch, chunk_seconds * SAMPLE_RATE, []
        for start in range(0, len(y), step):
            inputs = self.processor(y[start:start + step], sampling_rate=SAMPLE_RATE, return_tensors="pt")
            with torch.inference_mode():
                hidden = self.model(inputs.input_values.to(self.device), output_hidden_states=True).hidden_states
            out.append(torch.stack([hidden[i][0] for i in layers]).float().mean(dim=0).cpu().numpy())
        return np.concatenate(out)

    def embed_file(self, path):
        return self(load_audio(path))


def extract_embeddings(paths, encoder_name, device=None, cache_dir=EMBED_DIR, progress=True):
    """Embed every file in `paths`, caching to `cache_dir/<encoder>.npy`.

    Work is checkpointed every CHUNK clips, so an interrupted run (e.g. a Colab
    disconnect) resumes where it stopped.
    """
    cache_dir = Path(cache_dir)
    names = [Path(p).name for p in paths]
    final = cache_dir / f"{encoder_name}.npy"
    index = cache_dir / f"{encoder_name}.json"
    if final.exists() and index.exists() and json.loads(index.read_text()) == names:
        return np.load(final)

    parts_dir = cache_dir / f"{encoder_name}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    encoder = None
    chunks = []
    starts = range(0, len(paths), CHUNK)
    if progress:
        from tqdm.auto import tqdm
        starts = tqdm(starts, desc=f"{encoder_name} ({CHUNK} clips/step)")
    for start in starts:
        part = parts_dir / f"{start:06d}_{len(paths)}.npy"
        if not part.exists():
            encoder = encoder or Encoder(encoder_name, device)
            batch = np.stack([encoder.embed_file(p) for p in paths[start:start + CHUNK]])
            np.save(part, batch)
        chunks.append(np.load(part))

    embeddings = np.concatenate(chunks)
    np.save(final, embeddings)
    index.write_text(json.dumps(names))
    for part in parts_dir.glob("*.npy"):
        part.unlink()
    parts_dir.rmdir()
    return embeddings


def make_probe(C=0.1):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def pool_layers(embeddings, layers):
    """Average the selected layers: (n, layers + 1, dim) -> (n, dim)."""
    return embeddings[:, list(layers)].astype(np.float32).mean(axis=1)


def _fold_accuracy(embeddings, layers, C, y, tr, va):
    X = pool_layers(embeddings, layers)
    clf = make_probe(C).fit(X[tr], y[tr])
    return accuracy_score(y[va], clf.predict(X[va]))


def grouped_cv_grid(embeddings, y, groups, configs, n_splits=4, n_jobs=-1):
    """Actor-grouped CV accuracy for each (layers, C) config; all fits run in parallel."""
    folds = list(GroupKFold(n_splits).split(y, y, groups))
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_fold_accuracy)(embeddings, layers, C, y, tr, va)
        for layers, C in configs for tr, va in folds
    )
    return np.asarray(scores).reshape(len(configs), n_splits).mean(axis=1).tolist()


def scan_layers(embeddings, y, groups, C=0.1, n_splits=4):
    """Actor-grouped CV accuracy of a linear probe on each layer separately."""
    configs = [((i,), C) for i in range(embeddings.shape[1])]
    return grouped_cv_grid(embeddings, y, groups, configs, n_splits)


class EmbeddingEmotionModel:
    """Encoder name + chosen layers + fitted probe, persisted without the encoder weights."""

    def __init__(self, encoder_name, layers, classifier):
        self.encoder_name, self.layers, self.classifier = encoder_name, list(layers), classifier
        self._encoder = None

    @property
    def classes_(self):
        return self.classifier.classes_

    def features(self, embeddings):
        return pool_layers(embeddings, self.layers)

    def predict_file(self, path, device=None):
        if self._encoder is None:
            self._encoder = Encoder(self.encoder_name, device)
        x = self.features(self._encoder.embed_file(path)[None])
        proba = self.classifier.predict_proba(x)[0]
        scores = dict(zip(self.classes_, proba.round(4).tolist()))
        return self.classes_[int(np.argmax(proba))], scores

    def __getstate__(self):
        return {**self.__dict__, "_encoder": None}

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
