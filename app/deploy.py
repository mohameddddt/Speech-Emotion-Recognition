"""Assemble the Hugging Face Space from app/ + ser/ and upload it.

    hf auth login                                   # once, with a write token
    python app/deploy.py                            # -> huggingface.co/spaces/mohameddddt/speech-emotion-recognition
    python app/deploy.py --dry-run                  # only build app/build/space/ (run it with python app/build/space/server.py)
"""
import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "build" / "space"
SPACE_FILES = ["server.py", "results.json", "requirements.txt", "Dockerfile"]


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "ser").mkdir(parents=True)
    for name in SPACE_FILES:
        shutil.copy(HERE / name, OUT / name)
    shutil.copy(HERE / "SPACE_README.md", OUT / "README.md")
    shutil.copytree(HERE / "models", OUT / "models")
    shutil.copytree(HERE / "web", OUT / "web")
    shutil.copytree(HERE / "examples", OUT / "examples")
    for src in (ROOT / "ser").glob("*.py"):
        (OUT / "ser" / src.name).write_text(src.read_text(encoding="utf-8-sig"), encoding="utf-8")
    return OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", default="mohameddddt/speech-emotion-recognition")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = build()
    print("built", out)
    if args.dry_run:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=out, repo_id=args.space, repo_type="space",
                      commit_message="Deploy from Speech-Emotion-Recognition")
    print(f"https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
