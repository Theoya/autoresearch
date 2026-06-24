"""
Data preparation for the 3-class EXERCISE CLASSIFIER (deadlift / squat / bench).

Reuses the already-extracted 334-feature parquets (sex + 75 frames x 334 MediaPipe+DensePose
features = 25,051 cols). Per-clip feature vector = multi-stat temporal aggregation over the 75
frames (same 7-stat scheme as the regression trainers). Label = exercise class.

Classes:  0 = deadlift, 1 = squat, 2 = bench

Sources (additive, originals untouched):
  deadlift : kinetics/labelled/claude_*.parquet                 (74)
  squat    : datasets/claude_squat_labelled/*.parquet           (57)
  bench    : datasets/claude_bench_labelled/*.parquet           (22)

Note: only the claude_ deadlift parquets are used (so all three classes come from the same
claude_dataset extraction pipeline / distribution — no domain-shift confound from the older
kinetics clip_* samples).

Usage: python prepare_classifier.py   # load, print class counts + sanity stats
"""

import os
import glob

import numpy as np
import pandas as pd
import torch

NUM_FRAMES = 75
NUM_FEATURES_PER_FRAME = 334
CLASS_NAMES = ["deadlift", "squat", "bench"]

SOURCES = [
    ("deadlift", 0, "T:/clipforge/datasets/kinetics/labelled", "claude_*.parquet"),
    ("squat",    1, "T:/clipforge/datasets/claude_squat_labelled", "*.parquet"),
    ("bench",    2, "T:/clipforge/datasets/claude_bench_labelled", "*.parquet"),
]

CACHE_DIR = ".cache"


def _read_clip_features(pq_path):
    """Return (75, 335) float32 array: 334 features + sex per frame (same layout as prepare.py)."""
    df = pd.read_parquet(pq_path)
    assert len(df) == 1, f"Expected 1 row, got {len(df)} in {pq_path}"
    row = df.iloc[0]
    sex_val = float(row["sex"])
    frame_features = []
    for frame_idx in range(1, NUM_FRAMES + 1):
        prefix = f"f{frame_idx:02d}_"
        frame_cols = [c for c in df.columns if c.startswith(prefix)]
        assert len(frame_cols) == NUM_FEATURES_PER_FRAME, \
            f"Frame {frame_idx} has {len(frame_cols)} features, expected {NUM_FEATURES_PER_FRAME}"
        vals = row[frame_cols].values.astype(np.float32)
        vals = np.append(vals, sex_val)
        frame_features.append(vals)
    return np.stack(frame_features)  # (75, 335)


def load_all_data():
    """Load every clip across the three classes.

    Returns:
        X: tensor (N, 75, 335)
        y: tensor (N,) int64 class labels
        ids: list of (class_name, sample_id)
    """
    X_list, y_list, ids = [], [], []
    for cls_name, cls_idx, data_dir, pattern in SOURCES:
        files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        for pq in files:
            sid = os.path.splitext(os.path.basename(pq))[0]
            X_list.append(_read_clip_features(pq))
            y_list.append(cls_idx)
            ids.append((cls_name, sid))
    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.long)
    return X, y, ids


def load_all_data_cached():
    """Cache raw load keyed on the full set of parquet filenames across the three dirs."""
    import hashlib
    all_files = []
    for _, _, data_dir, pattern in SOURCES:
        all_files += sorted(glob.glob(os.path.join(data_dir, pattern)))
    key = hashlib.md5("|".join(os.path.basename(f) for f in all_files).encode()).hexdigest()[:12]
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f"clf_{len(all_files)}_{key}.pt")
    if os.path.exists(cp):
        b = torch.load(cp)
        return b["X"], b["y"], b["ids"]
    X, y, ids = load_all_data()
    torch.save({"X": X, "y": y, "ids": ids}, cp)
    return X, y, ids


def aggregate(X):
    """(N, 75, 335) -> (N, 2345) via mean/std/min/max/median/q25/q75 across frames."""
    parts = [X.mean(dim=1), X.std(dim=1), X.min(dim=1).values, X.max(dim=1).values,
             X.median(dim=1).values, X.quantile(0.25, dim=1), X.quantile(0.75, dim=1)]
    return torch.cat(parts, dim=1)


if __name__ == "__main__":
    X, y, ids = load_all_data_cached()
    print(f"Loaded {len(ids)} clips")
    print(f"X shape: {X.shape}   y shape: {y.shape}")
    print("\nClass counts:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i} {name:9s}: {(y == i).sum().item()}")
    flat = X.reshape(-1, X.shape[-1])
    print(f"\nNaN: {torch.isnan(flat).sum().item()}  Inf: {torch.isinf(flat).sum().item()}")
    Xa = aggregate(X)
    print(f"Aggregated feature dim: {Xa.shape[1]}")
