"""Train the emotion classifier, evaluate it on held-out actors and save it.

    python train.py            # train on 80% of actors, report test metrics, save model
    python train.py --full     # additionally refit on every actor before saving
"""
import argparse
import json

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from ser.data import ARTIFACTS_DIR, load_features, speaker_split, xy
from ser.model import MODEL_PATH, build_model, save_model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="refit on all data before saving")
    parser.add_argument("--refresh", action="store_true", help="re-extract features instead of using the cache")
    args = parser.parse_args()

    df = load_features(refresh=args.refresh, verbose=5)
    train, test = speaker_split(df)
    X_train, y_train = xy(train)
    X_test, y_test = xy(test)
    print(f"train: {len(train)} clips / {train.actor.nunique()} actors, "
          f"test: {len(test)} clips / {test.actor.nunique()} actors")

    model = build_model().fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
        "labels": model.classes_.tolist(),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=model.classes_).tolist(),
        "trained_on": "all actors" if args.full else "train actors",
    }
    print(classification_report(y_test, pred, digits=3))

    if args.full:
        X_all, y_all = xy(df)
        model = build_model().fit(X_all, y_all)

    save_model(model)
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"accuracy={metrics['accuracy']} macro_f1={metrics['macro_f1']} -> saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
