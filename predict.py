"""Predict the emotion of one or more audio files.

    python predict.py data/AudioWAV/1001_DFA_ANG_XX.wav path/to/other.wav
    python predict.py --model artifacts/embed_wavlm-base-plus.joblib my_recording.wav
"""
import argparse

from ser.embeddings import EmbeddingEmotionModel
from ser.model import MODEL_PATH, load_model, predict_file


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="audio files to classify")
    parser.add_argument("--model", default=MODEL_PATH,
                        help="saved model: the hand-crafted ensemble (default) or an embed_*.joblib probe")
    args = parser.parse_args()

    model = load_model(args.model)
    for path in args.files:
        if isinstance(model, EmbeddingEmotionModel):
            label, scores = model.predict_file(path)
        else:
            label, scores = predict_file(model, path)
        ranked = ", ".join(f"{k} {v:.2f}" for k, v in sorted(scores.items(), key=lambda kv: -kv[1]))
        print(f"{path}: {label}  ({ranked})")


if __name__ == "__main__":
    main()
