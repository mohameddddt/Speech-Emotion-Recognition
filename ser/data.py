"""Feature caching and speaker-independent splits."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ser.features import build_metadata, extract_dataset, feature_names

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "AudioWAV"
CACHE = ROOT / "artifacts" / "features.pkl"


def load_features(refresh=False, verbose=0):
    """Return a DataFrame with metadata columns followed by one column per feature."""
    if CACHE.exists() and not refresh:
        return pd.read_pickle(CACHE)
    meta = build_metadata(DATA_DIR)
    X = extract_dataset(meta["path"], verbose=verbose)
    df = pd.concat([meta, pd.DataFrame(X, columns=feature_names())], axis=1)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(CACHE)
    return df


def speaker_split(df, test_size=0.2, seed=42):
    """Split so that no actor appears in both train and test."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["actor"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def xy(df):
    return df[feature_names()].to_numpy(np.float32), df["emotion"].to_numpy()
