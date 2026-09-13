"""Train a linear probe on frozen pretrained speech embeddings and compare with the baseline.

    python train_embeddings.py                              # WavLM-base-plus
    python train_embeddings.py --encoder hubert-base
    python train_embeddings.py --encoder wavlm-large --full # refit on all actors before saving

Layer choice and regularisation are selected with actor-grouped CV on the training actors;
the held-out actors (same split as train.py) are used once for the final score.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from ser.data import ROOT, DATA_DIR, speaker_split
from ser.embeddings import (ENCODERS, EmbeddingEmotionModel, extract_embeddings, grouped_cv_grid,
                            make_probe, pool_layers, scan_layers)
from ser.features import build_metadata

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3]


def select_probe(E, y, groups, verbose=True):
    """Pick the layer window and C for the probe using grouped CV on training data only."""
    layer_scores = scan_layers(E, y, groups)
    best = int(np.argmax(layer_scores))
    if verbose:
        print("per-layer CV accuracy:", " ".join(f"{i}:{s:.3f}" for i, s in enumerate(layer_scores)))

    n_layers = E.shape[1]
    windows = {(best,), tuple(range(1, n_layers))}  # best single layer, all transformer layers
    for w in (1, 2, 3, 4):
        windows.add(tuple(range(max(0, best - w), min(n_layers, best + w + 1))))
    configs = [(layers, C) for layers in sorted(windows, key=len) for C in C_GRID]
    scores = grouped_cv_grid(E, y, groups, configs)
    score, (layers, C) = max(zip(scores, configs), key=lambda c: c[0])
    if verbose:
        print(f"selected layers={list(layers)} C={C} (CV accuracy {score:.3f})")
    return {"layer_scores": layer_scores, "layers": list(layers), "C": C, "cv_accuracy": score}


def run(encoder, device=None, full=False, verbose=True):
    meta = build_metadata(DATA_DIR)
    E = extract_embeddings(list(meta["path"]), encoder, device=device, progress=verbose)
    meta["row"] = np.arange(len(meta))
    train, test = speaker_split(meta)
    tr, te = train["row"].to_numpy(), test["row"].to_numpy()
    y = meta["emotion"].to_numpy()

    sel = select_probe(E[tr], y[tr], meta["actor"].to_numpy()[tr], verbose)
    X = pool_layers(E, sel["layers"])
    clf = make_probe(sel["C"]).fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    metrics = {
        "encoder": encoder,
        "accuracy": round(accuracy_score(y[te], pred), 4),
        "macro_f1": round(f1_score(y[te], pred, average="macro"), 4),
        **{k: sel[k] for k in ("layers", "C", "cv_accuracy")},
        "layer_cv_scores": [round(s, 4) for s in sel["layer_scores"]],
        "labels": clf.classes_.tolist(),
        "confusion_matrix": confusion_matrix(y[te], pred, labels=clf.classes_).tolist(),
        "trained_on": "all actors" if full else "train actors",
    }
    if verbose:
        print(classification_report(y[te], pred, digits=3))

    if full:
        clf = make_probe(sel["C"]).fit(X, y)
    model = EmbeddingEmotionModel(encoder, sel["layers"], clf)
    out = ROOT / "artifacts" / f"embed_{encoder}"
    model.save(out.with_suffix(".joblib"))
    out.with_suffix(".json").write_text(json.dumps(metrics, indent=2))

    baseline = ROOT / "artifacts" / "metrics.json"
    if verbose:
        msg = f"{encoder}: accuracy={metrics['accuracy']} macro_f1={metrics['macro_f1']}"
        if baseline.exists():
            msg += f"  (hand-crafted baseline: {json.loads(baseline.read_text())['accuracy']})"
        print(msg, "->", out.with_suffix(".joblib").relative_to(ROOT))
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default="wavlm-base-plus",
                        help=f"one of {list(ENCODERS)} or any Hugging Face model id")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--full", action="store_true", help="refit the probe on all actors before saving")
    args = parser.parse_args()
    run(args.encoder, args.device, args.full)


if __name__ == "__main__":
    main()
