"""Model definition, persistence and inference."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ser.data import ARTIFACTS_DIR
from ser.features import extract_file

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"


def build_model(seed=0):
    """Soft-voting ensemble; hyperparameters were chosen with actor-grouped CV (see notebook)."""
    return VotingClassifier(
        estimators=[
            ("logreg", make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=3000))),
            ("svm", make_pipeline(StandardScaler(), SVC(C=10, gamma=0.0005, probability=True, random_state=seed))),
            ("mlp", make_pipeline(
                StandardScaler(),
                MLPClassifier((512, 256), alpha=1e-2, early_stopping=True, max_iter=300, random_state=seed),
            )),
        ],
        voting="soft",
        n_jobs=-1,
    )


def save_model(model, path=MODEL_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict_file(model, wav_path):
    """Return (predicted emotion, {emotion: probability}) for one audio file."""
    proba = model.predict_proba(extract_file(wav_path)[None, :])[0]
    scores = dict(zip(model.classes_, proba.round(4).tolist()))
    return model.classes_[int(np.argmax(proba))], scores
