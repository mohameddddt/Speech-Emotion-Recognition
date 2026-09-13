"""Predict the emotion of one or more WAV files.

    python predict.py data/AudioWAV/1001_DFA_ANG_XX.wav path/to/other.wav
"""
import argparse

from ser.model import load_model, predict_file


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="audio files to classify")
    args = parser.parse_args()

    model = load_model()
    for path in args.files:
        label, scores = predict_file(model, path)
        ranked = ", ".join(f"{k} {v:.2f}" for k, v in sorted(scores.items(), key=lambda kv: -kv[1]))
        print(f"{path}: {label}  ({ranked})")


if __name__ == "__main__":
    main()
