"""Load a portable probe once and classify raw audio arrays (used by the web app).

    predictor = Predictor("app/models/wavlm-base-plus.npz")
    predictor.predict(y, sr)            # {"angry": 0.91, "disgust": 0.04, ...}
    predictor.predict_timeline(y, sr)   # per-window scores for longer recordings
"""
import librosa
import numpy as np

from ser.features import SAMPLE_RATE, extract_features, preprocess, to_mono_16k
from ser.probe import LinearProbe

MAX_SECONDS = 60
MIN_SPEECH_SECONDS = 0.3


class NoSpeechError(ValueError):
    pass


class Predictor:
    def __init__(self, probe, device=None):
        self.probe = probe if isinstance(probe, LinearProbe) else LinearProbe.load(probe)
        self.name = self.meta.get("encoder") or self.meta["features"]
        self.encoder = None
        if self.meta["features"] == "embedding":
            from ser.embeddings import Encoder  # torch/transformers only when needed
            self.encoder = Encoder(self.meta.get("hf_id", self.meta["encoder"]), device)
        self._features(np.zeros(SAMPLE_RATE, np.float32))  # warm-up: first call is slow otherwise

    @property
    def meta(self):
        return self.probe.meta

    @property
    def classes(self):
        return self.probe.classes_.tolist()

    def _features(self, y):
        if self.encoder is None:
            return extract_features(y)[None]
        hidden = self.encoder(y)  # (layers + 1, dim)
        return hidden[self.meta["layers"]].astype(np.float32).mean(axis=0)[None]

    def _scores(self, y16):
        proba = self.probe.predict_proba(self._features(preprocess(y16)))[0]
        return dict(zip(self.classes, proba.round(4).tolist()))

    def _window_scores(self, y16, voiced, chunks):
        """Encode the whole clip once, then pool each window's voiced frames.

        Equivalent to the training-time pooling (time and layer means commute) without re-running
        the encoder for every overlapping window: one pass instead of ~len(chunks).
        """
        hop = SAMPLE_RATE // 50  # one encoder frame = 20 ms
        F = self.encoder.frames(y16 / max(float(np.abs(y16).max()), 1e-6), self.meta["layers"])
        n = len(F)
        frame_voiced = np.array([voiced[i * hop:(i + 1) * hop].mean() > 0.5 for i in range(n)])
        X = []
        for c in chunks:
            f0, f1 = c.start // hop, min(n, -(-c.stop // hop))
            keep = frame_voiced[f0:f1]
            X.append(F[f0:f1][keep].mean(axis=0) if keep.any() else F[f0:f1].mean(axis=0))
        proba = self.probe.predict_proba(np.stack(X))
        return [dict(zip(self.classes, p.round(4).tolist())) for p in proba]

    def _prepare(self, y, sr):
        y16 = to_mono_16k(y, sr)[: MAX_SECONDS * SAMPLE_RATE]
        speech = librosa.effects.split(y16, top_db=30) if np.abs(y16).max() > 1e-3 else np.empty((0, 2))
        if sum(e - s for s, e in speech) < MIN_SPEECH_SECONDS * SAMPLE_RATE:
            raise NoSpeechError("No speech detected. Try recording closer to the microphone.")
        return y16, speech

    def predict(self, y, sr):
        y16, _ = self._prepare(y, sr)
        return self._scores(y16)

    def predict_timeline(self, y, sr, window=3.0, hop=1.0):
        """Scores for overlapping windows that contain speech, plus their average.

        Returns (overall_scores, [(start_s, end_s, scores), ...]). Clips shorter than
        1.5 windows are scored in one piece and give a single segment.
        """
        y16, speech = self._prepare(y, sr)
        total = len(y16) / SAMPLE_RATE
        if total <= 1.5 * window:
            scores = self._scores(y16)
            return scores, [(0.0, round(total, 2), scores)]

        voiced = np.zeros(len(y16), bool)
        for s, e in speech:
            voiced[s:e] = True
        win, step = int(window * SAMPLE_RATE), int(hop * SAMPLE_RATE)
        chunks = [slice(start, min(start + win, len(y16)))
                  for start in range(0, max(1, len(y16) - win + step), step)]
        chunks = [c for c in chunks if voiced[c].mean() >= 0.4]
        if self.encoder is None:
            scores = [self._scores(y16[c]) for c in chunks]
        else:
            scores = self._window_scores(y16, voiced, chunks)
        segments = [(round(c.start / SAMPLE_RATE, 2), round(c.stop / SAMPLE_RATE, 2), sc)
                    for c, sc in zip(chunks, scores)]
        if not segments:
            scores = self._scores(y16)
            return scores, [(0.0, round(total, 2), scores)]
        overall = {c: round(float(np.mean([s[c] for _, _, s in segments])), 4) for c in self.classes}
        return overall, segments
