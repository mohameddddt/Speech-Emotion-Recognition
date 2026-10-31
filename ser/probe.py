"""Portable linear probe: StandardScaler + multinomial logistic regression stored as plain arrays.

A probe file (`.npz`) holds the scaler and classifier weights plus a JSON `meta` string that says
which features it expects:

    {"features": "embedding", "encoder": "wavlm-base-plus", "layers": [5, 6, 7, 8, 9], ...}
    {"features": "handcrafted", ...}

No pickle is involved, so a probe can be loaded safely and without matching library versions,
and the maths (one affine map + softmax) can be reproduced in any language.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


class LinearProbe:
    def __init__(self, mean, scale, coef, intercept, classes, meta=None, temperature=1.0):
        self.mean = np.asarray(mean, np.float32)
        self.scale = np.asarray(scale, np.float32)
        self.coef = np.asarray(coef, np.float32)
        self.intercept = np.asarray(intercept, np.float32)
        self.classes_ = np.asarray(classes)
        self.meta = dict(meta or {})
        self.temperature = float(temperature)

    @classmethod
    def from_pipeline(cls, pipeline, meta=None, temperature=1.0):
        """Convert a fitted `make_pipeline(StandardScaler(), LogisticRegression())`."""
        scaler, clf = pipeline[0], pipeline[-1]
        return cls(scaler.mean_, scaler.scale_, clf.coef_, clf.intercept_, clf.classes_, meta, temperature)

    def logits(self, X):
        X = (np.atleast_2d(np.asarray(X, np.float32)) - self.mean) / self.scale
        return X @ self.coef.T + self.intercept

    def predict_proba(self, X, calibrated=True):
        z = self.logits(X)
        return softmax(z / (self.temperature if calibrated else 1.0), axis=1)

    def predict(self, X):
        return self.classes_[self.logits(X).argmax(axis=1)]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {**self.meta, "temperature": self.temperature}
        np.savez(path, mean=self.mean, scale=self.scale, coef=self.coef, intercept=self.intercept,
                 classes=self.classes_.astype(str), meta=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            return cls(f["mean"], f["scale"], f["coef"], f["intercept"], f["classes"],
                       meta, meta.get("temperature", 1.0))


def fit_temperature(logits, y, classes):
    """Single temperature minimising the NLL of held-out logits (Guo et al., 2017)."""
    idx = np.searchsorted(np.asarray(classes), y)

    def nll(t):
        return -log_softmax(logits / t, axis=1)[np.arange(len(y)), idx].mean()

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)
