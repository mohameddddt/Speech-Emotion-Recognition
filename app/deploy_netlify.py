"""Build the website as a static Netlify site, optionally proxying live analysis to a model backend.

    python app/deploy_netlify.py                                   # build app/build/netlify/ only
    python app/deploy_netlify.py --backend https://<space>.hf.space # proxy /api/predict to the backend
    python app/deploy_netlify.py --deploy                          # draft deploy (preview URL)
    python app/deploy_netlify.py --deploy --prod                   # production deploy

Netlify can't run the 95 M-parameter encoder, so the page, chart data and example clips are static,
and the six example clips ship with results precomputed for every model: they work with no backend.
With --backend, Netlify proxies /api/predict to the Docker server (app/server.py, e.g. the Hugging
Face Space from app/deploy.py), so recordings and uploads are analysed live on the same origin.

Deploying needs the Netlify CLI, logged in once: `npx netlify-cli login`.
"""
import argparse
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "build" / "netlify"
sys.path.insert(0, str(HERE))


def precompute(examples, models):
    """Run every model on every example clip, in the same format as the server's /api/predict."""
    import soundfile as sf

    from server import MODELS  # noqa: E402  (also puts `ser` on the path)
    from ser.inference import Predictor

    for m in models:  # one encoder in memory at a time: WavLM Large alone needs ~1.3 GB
        p = Predictor(MODELS[m["id"]]["path"])
        for ex in examples:
            y, sr = sf.read(HERE / ex["url"].lstrip("/"), dtype="float32")
            t = time.perf_counter()
            overall, segments = p.predict_timeline(y, sr)
            ex.setdefault("results", {})[m["id"]] = {
                "model": m["id"], "scores": overall, "duration": round(len(y) / sr, 2),
                "segments": [{"start": a, "end": b, "scores": s} for a, b, s in segments],
                "latency_ms": round((time.perf_counter() - t) * 1000)}
        print(f"precomputed {len(examples)} examples with {m['id']}", flush=True)
        del p
        gc.collect()


# `netlify deploy --dir` ignores a netlify.toml inside the deployed folder; _redirects and _headers
# files in the publish directory are always applied.
def redirects(backend):
    lines = ["/api/config   /api/config.json   200", "/api/results  /api/results.json  200"]
    if backend:  # proxy: the browser only ever talks to the Netlify origin, so no CORS is needed
        lines.append(f"/api/predict  {backend.rstrip('/')}/api/predict  200!")
    return "\n".join(lines) + "\n"


HEADERS = """/*
  X-Content-Type-Options: nosniff
  Referrer-Policy: strict-origin-when-cross-origin
  Permissions-Policy: microphone=(self)
/examples/*
  Cache-Control: public, max-age=86400
"""


def build(backend, only=None):
    from server import config, results  # noqa: E402

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "api").mkdir(parents=True)
    shutil.copy(HERE / "web" / "index.html", OUT / "index.html")
    shutil.copytree(HERE / "web", OUT / "static", ignore=shutil.ignore_patterns("index.html"))
    shutil.copytree(HERE / "examples", OUT / "examples")

    cfg = config()
    if only:  # the demo offers only these models (their sample results are precomputed below)
        cfg["models"] = [m for m in cfg["models"] if m["id"] in only]
        if cfg["default_model"] not in only:
            cfg["default_model"] = cfg["models"][0]["id"]
    precompute(cfg["examples"], cfg["models"])
    cfg["backend"] = bool(backend)
    (OUT / "api" / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (OUT / "api" / "results.json").write_text(json.dumps(json.loads(results().body)), encoding="utf-8")
    (OUT / "_redirects").write_text(redirects(backend), encoding="utf-8")
    (OUT / "_headers").write_text(HEADERS, encoding="utf-8")
    return OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", help="model server base URL, e.g. https://user-space.hf.space")
    parser.add_argument("--deploy", action="store_true", help="deploy with the Netlify CLI")
    parser.add_argument("--prod", action="store_true", help="production deploy (default: draft preview)")
    parser.add_argument("--site", help="Netlify site name or id (default: the one linked in app/build/netlify)")
    parser.add_argument("--models", nargs="+", help="limit the demo to these model ids (default: all in app/models)")
    args = parser.parse_args()

    out = build(args.backend, args.models)
    print("built", out, "| live analysis:", args.backend or "off (samples only)")
    if args.deploy:
        cmd = ["npx", "--yes", "netlify-cli", "deploy", "--dir", str(out), "--message", "Speech Emotion Recognition"]
        if args.prod:
            cmd.append("--prod")
        if args.site:
            cmd += ["--site", args.site]
        subprocess.run(cmd, check=True, shell=sys.platform == "win32")


if __name__ == "__main__":
    main()
