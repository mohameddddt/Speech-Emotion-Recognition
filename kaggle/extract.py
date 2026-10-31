"""Kaggle GPU job: cache per-layer mean-pooled embeddings of every CREMA-D clip for each encoder.

Output (/kaggle/working/embeddings/<encoder>.npy + .json) is attached to the `evaluate` job,
so probes and CV can run on a CPU kernel without spending GPU quota.
"""
import glob
import json
import os
import time

WORK = "/kaggle/working"
DATA = next(p for p in glob.glob("/kaggle/input/**/AudioWAV", recursive=True) if os.path.isdir(p))
os.environ.update(SER_DATA_DIR=DATA, SER_ARTIFACTS_DIR=WORK, SER_EMBED_DIR=f"{WORK}/embeddings")

import torch  # noqa: E402

from ser.data import DATA_DIR  # noqa: E402
from ser.embeddings import ENCODERS, extract_embeddings  # noqa: E402
from ser.features import build_metadata  # noqa: E402

print("audio:", DATA_DIR, "| cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
meta = build_metadata(DATA_DIR)
paths = list(meta["path"])
print(len(paths), "clips")

log = {}
for name in ENCODERS:
    t = time.time()
    E = extract_embeddings(paths, name)
    log[name] = {"shape": list(E.shape), "seconds": round(time.time() - t)}
    print(name, log[name], flush=True)
    with open(f"{WORK}/extract_log.json", "w") as f:
        json.dump(log, f, indent=2)
