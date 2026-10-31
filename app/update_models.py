"""Write app/results.json (the data behind the website's charts) and install the website's models.

    python app/update_models.py                 # download the `ser-evaluate` Kaggle output, install all of it
    python app/update_models.py --no-download   # reuse artifacts/kaggle/
    python app/update_models.py --legacy        # rebuild results.json from the local original-split artifacts

With Kaggle output, every number is nested 5-fold actor-grouped CV and app/models/ is replaced
by the all-actor probes. --legacy leaves app/models/ untouched.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
KERNEL = "mohameddddt/ser-evaluate"
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
NAMES = {
    "wavlm-large": "WavLM Large", "wavlm-base-plus": "WavLM Base+", "hubert-base": "HuBERT Base",
    "wav2vec2-base": "wav2vec 2.0 Base", "handcrafted": "Hand-crafted · logistic regression",
    "handcrafted-ensemble": "Hand-crafted · ensemble",
}
# Original 72/19-actor split, from notebooks/project_notebook.ipynb (models not re-run on Kaggle).
LEGACY_HANDCRAFTED = [
    ("rf", "Hand-crafted · random forest", 0.469, 0.449),
    ("mlp", "Hand-crafted · MLP", 0.515, 0.516),
    ("svm", "Hand-crafted · SVM", 0.516, 0.511),
    ("handcrafted", "Hand-crafted · logistic regression", 0.522, 0.519),
    ("xgb", "Hand-crafted · XGBoost", 0.523, 0.517),
]
COMMON = {"chance": 1 / 6, "human": 0.409,
          "dataset": {"clips": 7442, "actors": 91, "sentences": 12, "emotions": 6}}
# Facts the page states about the deployed encoder (pretraining data from the model cards).
ENCODERS = {
    "wavlm-large": {"name": "WavLM Large", "hf_id": "microsoft/wavlm-large", "params": "317 M", "hours": "94k",
                    "layers": 24, "dim": 1024},
    "wavlm-base-plus": {"name": "WavLM Base+", "hf_id": "microsoft/wavlm-base-plus", "params": "95 M",
                        "hours": "94k", "layers": 12, "dim": 768},
    "hubert-base": {"name": "HuBERT Base", "hf_id": "facebook/hubert-base-ls960", "params": "95 M", "hours": "960",
                    "layers": 12, "dim": 768},
    "wav2vec2-base": {"name": "wav2vec 2.0 Base", "hf_id": "facebook/wav2vec2-base", "params": "95 M",
                      "hours": "960", "layers": 12, "dim": 768},
}


def encoder_info(name):
    info = dict(ENCODERS[name])
    info["probe_params"] = 6 * info["dim"] + 6
    return info


def actor_splits(protocol):
    from sklearn.model_selection import GroupKFold

    from ser.data import DATA_DIR, speaker_split
    from ser.features import build_metadata

    meta = build_metadata(DATA_DIR)
    if protocol == "legacy":
        return sorted(meta.actor.unique().tolist()), [sorted(speaker_split(meta)[1].actor.unique().tolist())]
    folds = GroupKFold(5).split(meta, groups=meta.actor)
    return (sorted(meta.actor.unique().tolist()),
            [sorted(meta.actor.iloc[te].unique().tolist()) for _, te in folds])


def roadmap(cv_done):
    return [
        {"status": "done", "title": "Speaker-independent evaluation",
         "text": "Test voices never appear in training. Model selection uses only training actors."},
        {"status": "done", "title": "Hand-crafted baseline · 53%",
         "text": "422 acoustic features (MFCC, pitch, energy, spectrum). Five classifiers plus an ensemble."},
        {"status": "done", "title": "Pretrained speech encoder · 74%",
         "text": "Frozen WavLM, a per-layer scan and a linear probe. +21 points from changing the features alone."},
        {"status": "done", "title": "Calibrated, portable models + this site",
         "text": "Probes stored as plain arrays, temperature-scaled, served on CPU in about a second."},
        {"status": "done" if cv_done else "running", "title": "Four encoders × 5-fold CV" + (" · 78%" if cv_done else ""),
         "text": "WavLM Base+/Large, HuBERT and wav2vec 2.0 under nested actor-grouped cross-validation, on Kaggle. "
                 + ("WavLM Large wins." if cv_done else "")},
        {"status": "next", "title": "Cross-corpus and real-microphone tests",
         "text": "RAVDESS, TESS and SAVEE without retraining, plus recordings from this site's own microphones."},
        {"status": "next", "title": "Fine-tuning with augmentation",
         "text": "End-to-end WavLM training with noise, reverberation and phone-mic simulation, for robustness outside the studio."},
    ]


def legacy_results():
    art = ROOT / "artifacts"
    base = json.loads((art / "metrics.json").read_text())
    emb = json.loads((art / "embed_wavlm-base-plus.json").read_text())
    models = [{"id": i, "name": n, "family": "hand-crafted", "accuracy": a, "macro_f1": f}
              for i, n, a, f in LEGACY_HANDCRAFTED]
    models.append({"id": "handcrafted-ensemble", "name": NAMES["handcrafted-ensemble"], "family": "hand-crafted",
                   "accuracy": base["accuracy"], "macro_f1": base["macro_f1"]})
    models.append({"id": "wavlm-base-plus", "name": NAMES["wavlm-base-plus"], "family": "pretrained",
                   "accuracy": emb["accuracy"], "macro_f1": emb["macro_f1"]})
    actors, splits = actor_splits("legacy")
    return {**COMMON, "protocol": "legacy", "encoder": encoder_info("wavlm-base-plus"),
            "protocol_text": "19 held-out actors · 1,552 clips", "best": "wavlm-base-plus", "models": models,
            "layers": {"model": NAMES["wavlm-base-plus"], "scores": emb["layer_cv_scores"], "selected": emb["layers"]},
            "confusion": {"labels": LABELS, "models": {
                "wavlm-base-plus": {"name": NAMES["wavlm-base-plus"], "matrix": emb["confusion_matrix"]},
                "handcrafted-ensemble": {"name": NAMES["handcrafted-ensemble"], "matrix": base["confusion_matrix"]}}},
            "actors": actors, "splits": splits, "roadmap": roadmap(False)}


def cv_results(summary):
    best = max(summary, key=lambda n: summary[n]["cv5"]["accuracy_mean"] if summary[n]["kind"] == "embedding" else -1)
    models = [{"id": n, "name": NAMES.get(n, n), "family": "pretrained" if r["kind"] == "embedding" else "hand-crafted",
               "accuracy": r["cv5"]["accuracy_mean"], "std": r["cv5"]["accuracy_std"],
               "macro_f1": r["cv5"]["macro_f1_mean"], "legacy_accuracy": r["legacy"]["accuracy"]}
              for n, r in summary.items()]
    confusion = {n: {"name": NAMES.get(n, n), "matrix": summary[n]["cv5"]["confusion_matrix"]}
                 for n in (best, "handcrafted-ensemble") if n in summary}
    if not any(m["family"] == "hand-crafted" for m in models):
        # The Kaggle hand-crafted stage didn't finish: show the original-split numbers, labelled as such.
        legacy = legacy_results()
        for m in legacy["models"]:
            if m["family"] == "hand-crafted":
                models.append({**m, "protocol": "legacy"})
        hc = legacy["confusion"]["models"]["handcrafted-ensemble"]
        confusion["handcrafted-ensemble"] = {**hc, "name": hc["name"] + " (original split)"}
    models.sort(key=lambda m: m["accuracy"])
    actors, splits = actor_splits("cv5")
    full = summary[best]["full"]
    return {**COMMON, "protocol": "cv5", "encoder": encoder_info(best),
            "protocol_text": "nested 5-fold actor-grouped CV · all 7,442 clips", "best": best, "models": models,
            "layers": {"model": NAMES.get(best, best), "scores": full["layer_cv_scores"], "selected": full["layers"]},
            "confusion": {"labels": LABELS, "models": confusion},
            "actors": actors, "splits": splits, "roadmap": roadmap(True)}


def write(results):
    (HERE / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("wrote app/results.json:", results["protocol"], "best", results["best"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="src", default=str(ROOT / "artifacts" / "kaggle"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--models", nargs="*", default=None, help="probe names to install (default: all)")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(ROOT))
    if args.legacy:
        return write(legacy_results())

    src = Path(args.src)
    if not args.no_download:
        src.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "kernels", "output", KERNEL, "-p", str(src), "-o",
                        "--file-pattern", r"(probe_.*\.npz|summary\.(json|md)|embed_.*\.json|.*\.log)"], check=True)
    summary = json.loads((src / "summary.json").read_text(encoding="utf-8"))
    probes = {p.stem.removeprefix("probe_"): p for p in src.glob("probe_*.npz")}
    chosen = args.models if args.models is not None else list(probes)
    for old in (HERE / "models").glob("*.npz"):
        old.unlink()
    for name in chosen:
        shutil.copy(probes[name], HERE / "models" / f"{name}.npz")
        print("installed", name)
    write(cv_results(summary))


if __name__ == "__main__":
    main()
