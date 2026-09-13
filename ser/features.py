"""Audio loading and hand-crafted feature extraction for speech emotion recognition."""
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SAMPLE_RATE = 16000
N_MFCC = 40
N_MELS = 64

EMOTIONS = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def parse_filename(path):
    """CREMA-D names look like `1001_DFA_ANG_XX.wav` (actor_sentence_emotion_intensity)."""
    actor, sentence, emotion, intensity = Path(path).stem.split("_")
    return {
        "path": str(path),
        "actor": int(actor),
        "sentence": sentence,
        "emotion": EMOTIONS[emotion],
        "intensity": intensity,
    }


def build_metadata(data_dir):
    files = sorted(Path(data_dir).glob("*.wav"))
    return pd.DataFrame([parse_filename(f) for f in files])


def load_audio(path, sr=SAMPLE_RATE):
    y, _ = librosa.load(path, sr=sr, mono=True)
    trimmed, _ = librosa.effects.trim(y, top_db=30)
    # Fall back to the raw signal if trimming removes nearly everything.
    y = trimmed if len(trimmed) > sr // 10 else y
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y


def _stats(matrix):
    """Mean and std over time for a (n_features, n_frames) matrix."""
    return np.concatenate([matrix.mean(axis=1), matrix.std(axis=1)])


def extract_features(y, sr=SAMPLE_RATE):
    """Return a fixed-length feature vector describing one utterance."""
    stft = np.abs(librosa.stft(y, n_fft=512, hop_length=160))
    power = stft**2

    # mfcc + deltas: timbre / texture of the voice and how it changes
    mel = librosa.feature.melspectrogram(S=power, sr=sr, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # chroma: pitch-class distribution
    chroma = librosa.feature.chroma_stft(S=power, sr=sr, n_fft=512, tuning=0.0)
    # spectral shape
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, n_fft=512, n_bands=5, fmin=100)
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)
    flatness = librosa.feature.spectral_flatness(S=stft)
    # energy and noisiness
    rms = librosa.feature.rms(S=stft, frame_length=512)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=160)

    # prosody: fundamental frequency over voiced frames
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=160)
    f0 = f0[(f0 > 60) & (f0 < 400)]
    if len(f0) > 1:
        pitch = [f0.mean(), f0.std(), f0.min(), f0.max(), np.median(f0)]
    else:
        pitch = [0.0] * 5

    return np.concatenate([
        _stats(mfcc),
        _stats(delta),
        _stats(delta2),
        _stats(log_mel),
        _stats(chroma),
        _stats(contrast),
        _stats(np.vstack([centroid, bandwidth, rolloff, flatness, rms, zcr])),
        pitch,
        [len(y) / sr],
    ]).astype(np.float32)


def feature_names():
    names = []
    blocks = [
        ("mfcc", N_MFCC), ("mfcc_delta", N_MFCC), ("mfcc_delta2", N_MFCC),
        ("logmel", N_MELS), ("chroma", 12), ("contrast", 6),
    ]
    for block, n in blocks:
        names += [f"{block}_{i}_{s}" for s in ("mean", "std") for i in range(n)]
    scalar = ["centroid", "bandwidth", "rolloff", "flatness", "rms", "zcr"]
    names += [f"{f}_{s}" for s in ("mean", "std") for f in scalar]
    names += ["f0_mean", "f0_std", "f0_min", "f0_max", "f0_median", "duration"]
    return names


def extract_file(path):
    return extract_features(load_audio(path))


def extract_dataset(paths, n_jobs=-1, verbose=0):
    feats = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(extract_file)(p) for p in paths)
    return np.vstack(feats)
