"""Fast checks that need neither the dataset nor a speech encoder (synthetic audio only)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ser.data import speaker_split
from ser.features import SAMPLE_RATE, extract_features, feature_names, parse_filename, preprocess
from ser.inference import NoSpeechError, Predictor
from ser.probe import LinearProbe, fit_temperature

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad"]


def tone(seconds=1.0, sr=SAMPLE_RATE, freq=220.0):
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def test_parse_filename():
    assert parse_filename("data/1001_IEO_HAP_HI.wav") == {
        "path": "data/1001_IEO_HAP_HI.wav", "actor": 1001, "sentence": "IEO", "emotion": "happy", "intensity": "HI"}


def test_feature_vector_length():
    assert len(extract_features(preprocess(tone()))) == len(feature_names()) == 422


def test_preprocess_handles_browser_audio():
    y = tone(1.0, sr=44100)
    stereo_int16 = (np.stack([y, y], axis=1) * 20000).astype(np.int16)
    out = preprocess(stereo_int16, 44100)
    assert out.ndim == 1 and out.dtype == np.float32
    assert abs(len(out) - SAMPLE_RATE) < SAMPLE_RATE * 0.05
    assert np.isclose(np.abs(out).max(), 1.0)


def test_speaker_split_is_actor_disjoint():
    df = pd.DataFrame({"actor": np.repeat(np.arange(1001, 1021), 5)})
    train, test = speaker_split(df)
    assert not set(train.actor) & set(test.actor)
    assert len(train) + len(test) == len(df)


@pytest.fixture
def probe(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 422)).astype(np.float32)
    y = np.repeat(CLASSES, 20)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)).fit(X, y)
    path = tmp_path / "handcrafted.npz"
    LinearProbe.from_pipeline(pipe, {"features": "handcrafted"}, temperature=1.5).save(path)
    return pipe, X, path


def test_probe_matches_sklearn_and_round_trips(probe):
    pipe, X, path = probe
    loaded = LinearProbe.load(path)
    assert loaded.meta["features"] == "handcrafted" and loaded.temperature == 1.5
    np.testing.assert_allclose(loaded.predict_proba(X, calibrated=False), pipe.predict_proba(X), atol=1e-5)
    assert (loaded.predict(X) == pipe.predict(X)).all()


def test_fit_temperature_recovers_scale():
    rng = np.random.default_rng(1)
    true = rng.normal(size=(4000, 6)) * 2
    y = np.array([rng.choice(6, p=np.exp(z) / np.exp(z).sum()) for z in true])
    assert fit_temperature(true * 3, np.asarray(CLASSES)[y], CLASSES) == pytest.approx(3, rel=0.15)


def test_predictor_on_synthetic_audio(probe):
    _, _, path = probe
    p = Predictor(path)
    scores = p.predict(tone(2.0), SAMPLE_RATE)
    assert list(scores) == CLASSES and sum(scores.values()) == pytest.approx(1, abs=1e-3)

    overall, segments = p.predict_timeline(tone(9.0), SAMPLE_RATE)
    assert len(segments) > 1 and sum(overall.values()) == pytest.approx(1, abs=1e-3)

    with pytest.raises(NoSpeechError):
        p.predict(np.zeros(SAMPLE_RATE * 2, np.int16), SAMPLE_RATE)
