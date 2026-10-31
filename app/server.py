"""Showcase website + live demo API.

    python app/server.py            # http://127.0.0.1:7860

The page (app/web/) records or decodes audio in the browser, converts it to 16 kHz mono WAV and
POSTs it to /api/predict. Charts read /api/results (app/results.json, written by
app/update_models.py from the Kaggle evaluation). Models are the portable probes in app/models/.
"""
import asyncio
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
for root in (HERE, HERE.parent):
    if (root / "ser").is_dir():
        sys.path.insert(0, str(root))
        break

import soundfile as sf  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from ser.inference import MAX_SECONDS, NoSpeechError, Predictor  # noqa: E402
from ser.probe import LinearProbe  # noqa: E402

MAX_BYTES = 20 * 1024 * 1024
MODEL_NAMES = {
    "wavlm-large": "WavLM Large",
    "wavlm-base-plus": "WavLM Base+",
    "hubert-base": "HuBERT Base",
    "wav2vec2-base": "wav2vec 2.0 Base",
    "handcrafted": "Hand-crafted",
}
DEFAULT_ORDER = ["wavlm-large", "wavlm-base-plus", "hubert-base", "wav2vec2-base", "handcrafted"]


def discover_models():
    found = {p.stem: p for p in sorted((HERE / "models").glob("*.npz"))}
    order = [n for n in DEFAULT_ORDER if n in found] + sorted(n for n in found if n not in DEFAULT_ORDER)
    models = []
    for name in order:
        meta = LinearProbe.load(found[name]).meta
        models.append({"id": name, "name": MODEL_NAMES.get(name, name),
                       "accuracy": meta.get("cv5_accuracy", meta.get("legacy_accuracy")),
                       "features": meta["features"], "calibrated": meta.get("temperature", 1.0) != 1.0,
                       "path": str(found[name])})
    return models


MODELS = {m["id"]: m for m in discover_models()}
DEFAULT_MODEL = next(iter(MODELS))
# One dedicated inference thread: requests queue up (the encoder already uses every core), and the
# model is warmed up on the same thread that serves it, so the first visitor isn't slow.
_worker = ThreadPoolExecutor(max_workers=1)


@lru_cache(maxsize=None)
def predictor(model_id):
    return Predictor(MODELS[model_id]["path"])


app = FastAPI(title="Speech Emotion Recognition", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=HERE / "web"), name="static")
app.mount("/examples", StaticFiles(directory=HERE / "examples"), name="examples")


@app.get("/")
def index():
    return FileResponse(HERE / "web" / "index.html")


@app.get("/api/config")
def config():
    examples = []
    for p in sorted((HERE / "examples").glob("*.wav")):
        emotion, actor, sentence = p.stem.split("_")
        examples.append({"url": f"/examples/{p.name}", "emotion": emotion, "actor": int(actor), "sentence": sentence})
    return {"models": [{k: m[k] for k in ("id", "name", "accuracy", "features", "calibrated")} for m in MODELS.values()],
            "default_model": DEFAULT_MODEL, "examples": examples, "max_seconds": MAX_SECONDS}


@app.get("/api/results")
def results():
    return JSONResponse(json.loads((HERE / "results.json").read_text(encoding="utf-8")))


def _classify(model, y, sr):
    return predictor(model).predict_timeline(y, sr)


@app.post("/api/predict")
async def predict(request: Request, model: str = DEFAULT_MODEL):
    if model not in MODELS:
        raise HTTPException(404, f"unknown model {model!r}")
    if int(request.headers.get("content-length", 0)) > MAX_BYTES:
        raise HTTPException(413, "Audio file too large (20 MB max).")
    body = await request.body()
    try:
        y, sr = sf.read(io.BytesIO(body), dtype="float32", always_2d=False)
    except Exception:
        raise HTTPException(415, "Could not decode the audio. Send WAV, FLAC or OGG.")
    t = time.perf_counter()
    try:
        overall, segments = await asyncio.get_running_loop().run_in_executor(_worker, _classify, model, y, sr)
    except NoSpeechError as e:
        raise HTTPException(422, str(e))
    return {"model": model, "scores": overall, "duration": round(len(y) / sr, 2),
            "segments": [{"start": a, "end": b, "scores": s} for a, b, s in segments],
            "latency_ms": round((time.perf_counter() - t) * 1000)}


if __name__ == "__main__":
    import uvicorn

    if "--download" in sys.argv:  # used at Docker build time: fetch every encoder, then exit
        for model_id in MODELS:
            predictor(model_id)
        sys.exit(0)
    _worker.submit(predictor, DEFAULT_MODEL).result()  # load and warm up before accepting requests
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 7860)))
