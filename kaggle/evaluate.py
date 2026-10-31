"""Kaggle CPU job: compare every encoder (and the hand-crafted features) and export website models.

For each feature set:
  * legacy   - the original 72/19-actor split, so numbers stay comparable with the README;
  * cv5      - nested 5-fold actor-grouped CV over all 91 actors (layers and C re-selected
               inside every outer fold), reported as mean +- std;
  * export   - probe refit on all actors, temperature-calibrated on out-of-fold logits, saved
               as a portable `probe_<name>.npz` for the website.

Embeddings come from the `ser-extract-embeddings` kernel output; nothing here needs a GPU.
"""
import faulthandler
import glob
import json
import os
import time
import traceback

try:
    import resource  # Linux (Kaggle); absent on Windows
except ImportError:
    resource = None

faulthandler.enable()  # a hard crash (segfault, abort) still leaves a stack trace in the log

# Environment overrides allow a local smoke test on a subset; on Kaggle the inputs are discovered.
WORK = os.environ.get("SER_ARTIFACTS_DIR", "/kaggle/working")
DATA = os.environ.get("SER_DATA_DIR") or next(
    p for p in glob.glob("/kaggle/input/**/AudioWAV", recursive=True) if os.path.isdir(p))
EMBED = os.environ.get("SER_EMBED_DIR") or os.path.dirname(
    glob.glob("/kaggle/input/**/embeddings/wavlm-base-plus.npy", recursive=True)[0])
os.environ.update(SER_DATA_DIR=DATA, SER_ARTIFACTS_DIR=WORK, SER_EMBED_DIR=EMBED)
T0 = time.time()


def log(msg):
    """Progress line with elapsed time and peak memory, also appended to progress.log in the output."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 if resource else 0
    line = f"[{time.time() - T0:7.0f}s | peak {rss:,.0f} MB] {msg}"
    print(line, flush=True)
    with open(f"{WORK}/progress.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


log(f"audio: {DATA} | embeddings: {EMBED} | cpus: {os.cpu_count()}")

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ser.data import load_features, speaker_split, xy  # noqa: E402
from ser.embeddings import ENCODERS, extract_embeddings, make_probe, pool_layers  # noqa: E402
from ser.features import build_metadata  # noqa: E402
from ser.model import build_model  # noqa: E402
from ser.probe import LinearProbe, fit_temperature  # noqa: E402
from train_embeddings import run, select_probe  # noqa: E402

N_FOLDS = 5
HANDCRAFTED_C = 0.03  # chosen by grouped CV in notebooks/project_notebook.ipynb


def scores(y, pred):
    return {"accuracy": round(accuracy_score(y, pred), 4),
            "macro_f1": round(f1_score(y, pred, average="macro"), 4)}


def summarise_cv(folds, y, oof, labels):
    acc = [f["accuracy"] for f in folds]
    cm = confusion_matrix(y, oof, labels=labels)
    return {"accuracy_mean": round(float(np.mean(acc)), 4), "accuracy_std": round(float(np.std(acc)), 4),
            "macro_f1_mean": round(float(np.mean([f["macro_f1"] for f in folds])), 4),
            "folds": folds, "labels": list(labels), "confusion_matrix": cm.tolist(),
            "recall": dict(zip(labels, (cm.diagonal() / cm.sum(axis=1)).round(4).tolist()))}


def oof_logits(make, X, y, groups):
    """Out-of-fold logits of a fixed configuration, for temperature calibration."""
    z = np.zeros((len(y), len(np.unique(y))), np.float32)
    for tr, te in GroupKFold(N_FOLDS).split(X, y, groups):
        z[te] = LinearProbe.from_pipeline(make().fit(X[tr], y[tr])).logits(X[te])
    return z


def export(pipeline, X, y, groups, make, meta, name):
    temperature = fit_temperature(oof_logits(make, X, y, groups), y, pipeline.classes_)
    probe = LinearProbe.from_pipeline(pipeline, meta, temperature)
    probe.save(f"{WORK}/probe_{name}.npz")
    return temperature


def evaluate_encoder(name, meta_df, results):
    E = extract_embeddings(list(meta_df["path"]), name, progress=False)
    y, groups = meta_df["emotion"].to_numpy(), meta_df["actor"].to_numpy()
    labels = sorted(set(y))
    t = time.time()

    _, legacy = run(name, verbose=False)

    folds, oof = [], np.empty(len(y), dtype=object)
    for k, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(E[:, 0], y, groups)):
        sel = select_probe(E[tr], y[tr], groups[tr], verbose=False)
        X = pool_layers(E, sel["layers"])
        oof[te] = make_probe(sel["C"]).fit(X[tr], y[tr]).predict(X[te])
        folds.append({**scores(y[te], oof[te]), "layers": sel["layers"], "C": sel["C"]})
        log(f"  {name} fold {k}: {folds[-1]}")

    sel = select_probe(E, y, groups, verbose=False)
    X = pool_layers(E, sel["layers"])
    pipe = make_probe(sel["C"]).fit(X, y)
    cv = summarise_cv(folds, y, oof, labels)
    meta = {"features": "embedding", "encoder": name, "hf_id": ENCODERS[name], "layers": sel["layers"],
            "C": sel["C"], "trained_on": "all 91 CREMA-D actors",
            "cv5_accuracy": cv["accuracy_mean"], "cv5_accuracy_std": cv["accuracy_std"],
            "legacy_accuracy": legacy["accuracy"]}
    temperature = export(pipe, X, y, groups, lambda: make_probe(sel["C"]), meta, name)

    results[name] = {"kind": "embedding", "legacy": {k: legacy[k] for k in ("accuracy", "macro_f1", "layers", "C")},
                     "cv5": cv, "full": {"layers": sel["layers"], "C": sel["C"], "temperature": round(temperature, 3),
                                         "layer_cv_scores": [round(s, 4) for s in sel["layer_scores"]]},
                     "dim": int(E.shape[2]), "n_layers": int(E.shape[1]), "seconds": round(time.time() - t)}
    log(f"{name}: legacy {legacy['accuracy']}  cv5 {cv['accuracy_mean']} +- {cv['accuracy_std']}")


def evaluate_handcrafted(results):
    t = time.time()
    log("hand-crafted: extracting features")
    df = load_features(verbose=1)
    log(f"hand-crafted: features {df.shape}")
    X, y = xy(df)
    groups, labels = df["actor"].to_numpy(), sorted(set(y))
    train, test = speaker_split(df)
    (Xtr, ytr), (Xte, yte) = xy(train), xy(test)

    def logreg():
        return make_pipeline(StandardScaler(), LogisticRegression(C=HANDCRAFTED_C, max_iter=3000))

    for name, make in {"handcrafted": logreg, "handcrafted-ensemble": build_model}.items():
        legacy = scores(yte, make().fit(Xtr, ytr).predict(Xte))
        folds, oof = [], np.empty(len(y), dtype=object)
        for tr, te in GroupKFold(N_FOLDS).split(X, y, groups):
            oof[te] = make().fit(X[tr], y[tr]).predict(X[te])
            folds.append(scores(y[te], oof[te]))
        results[name] = {"kind": "handcrafted", "legacy": legacy, "cv5": summarise_cv(folds, y, oof, labels)}
        log(f"{name}: legacy {legacy['accuracy']}  cv5 {results[name]['cv5']['accuracy_mean']}")

    cv = results["handcrafted"]["cv5"]
    meta = {"features": "handcrafted", "C": HANDCRAFTED_C, "trained_on": "all 91 CREMA-D actors",
            "cv5_accuracy": cv["accuracy_mean"], "cv5_accuracy_std": cv["accuracy_std"],
            "legacy_accuracy": results["handcrafted"]["legacy"]["accuracy"]}
    temperature = export(logreg().fit(X, y), X, y, groups, logreg, meta, "handcrafted")
    results["handcrafted"]["full"] = {"C": HANDCRAFTED_C, "temperature": round(temperature, 3)}
    results["handcrafted"]["seconds"] = round(time.time() - t)


def write_summary(results):
    with open(f"{WORK}/summary.json", "w") as f:
        json.dump(results, f, indent=2)
    rows = sorted(results.items(), key=lambda kv: -kv[1]["cv5"]["accuracy_mean"])
    lines = ["| model | 5-fold actor CV accuracy | macro F1 | legacy split accuracy |", "|---|---|---|---|"]
    for name, r in rows:
        cv = r["cv5"]
        lines.append(f"| {name} | {cv['accuracy_mean']:.3f} ± {cv['accuracy_std']:.3f} | "
                     f"{cv['macro_f1_mean']:.3f} | {r['legacy']['accuracy']:.3f} |")
    with open(f"{WORK}/summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("summary:\n" + "\n".join(lines))


def stage(name, fn, *args):
    """Run one evaluation; a failure is logged with its traceback and the job moves on."""
    log(f"start {name}")
    try:
        fn(*args)
        write_summary(results)
    except Exception:
        log(f"FAILED {name}:\n{traceback.format_exc()}")


results = {}
meta_df = build_metadata(DATA)
log(f"{len(meta_df)} clips")
for name in ENCODERS:  # the embedding results matter most, so they run first
    if os.path.exists(os.path.join(EMBED, f"{name}.npy")):
        stage(name, evaluate_encoder, name, meta_df, results)
    else:
        log(f"no embeddings for {name}")
stage("hand-crafted", evaluate_handcrafted, results)
log("done")
