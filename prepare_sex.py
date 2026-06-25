"""
Data preparation for the 3-class SEX/GENDER CLASSIFIER (male / female / unclear).

Same recipe as prepare_classifier.py (the exercise classifier), but:
  * Label = the `gender` field Claude Code assigned in the dataset manifests
    (T:/clipforge/datasets/claude_dataset/{deadlift,squat,bench}/manifest.json),
    NOT the exercise class.
  * Features = the already-extracted pose columns in the parquets. We use the
    MediaPipe + limb + angle channels (mp_/limb_/angle_, 102 feats/frame) by default
    and SKIP the DensePose channels (cse_/px_), which the exercise-classifier ablation
    showed add noise. Body-proportion signal (shoulder/hip width, limb ratios) lives in
    the limb_/angle_ groups, so this is the plausible-signal subset anyway.
  * The parquet's own `sex` column is NOT used as a feature here -- predicting sex from
    a `sex` input would be leakage. (It is also an unset placeholder == 2.0 in most rows.)

Per-clip feature vector = multi-stat temporal aggregation over the 75 frames
(mean/std/min/max/median/q25/q75), identical to the exercise classifier and the
regression trainers.

Classes:  0 = male, 1 = female, 2 = unclear

Sources (additive, originals untouched). Clip id is the parquet stem with the
claude_ / claude_sweep_ prefix stripped; it joins 1:1 to manifest `clips[].id`:
  deadlift : kinetics/labelled/claude_*.parquet  (+ claude_sweep_*.parquet)
  squat    : datasets/claude_squat_labelled/*.parquet
  bench    : datasets/claude_bench_labelled/*.parquet

Usage: python prepare_sex.py   # load, print class counts + sanity stats
"""

import os
import glob
import json

import numpy as np
import pandas as pd
import torch

NUM_FRAMES = 75
NUM_FEATURES_PER_FRAME = 334
CLASS_NAMES = ["male", "female", "unclear"]
GENDER_TO_IDX = {"male": 0, "female": 1, "unclear": 2}

# (exercise, parquet_dir, glob_pattern, manifest_path)
SOURCES = [
    ("deadlift", "T:/clipforge/datasets/kinetics/labelled", "claude_*.parquet",
     "T:/clipforge/datasets/claude_dataset/deadlift/manifest.json"),
    ("squat", "T:/clipforge/datasets/claude_squat_labelled", "*.parquet",
     "T:/clipforge/datasets/claude_dataset/squat/manifest.json"),
    ("bench", "T:/clipforge/datasets/claude_bench_labelled", "*.parquet",
     "T:/clipforge/datasets/claude_dataset/bench/manifest.json"),
]

# MediaPipe-only = mp/limb/angle (pose-derivable, no Detectron2/GPU). DensePose = cse/px.
MP_GROUPS = ("mp_", "limb_", "angle_")
DP_GROUPS = ("cse_", "px_")

CACHE_DIR = ".cache"


def _clip_id_from_path(pq_path):
    """Parquet stem -> manifest clip id (strip claude_sweep_ / claude_ prefix)."""
    b = os.path.splitext(os.path.basename(pq_path))[0]
    if b.startswith("claude_sweep_"):
        return b[len("claude_sweep_"):]
    if b.startswith("claude_"):
        return b[len("claude_"):]
    return b


def _load_gender_map(manifest_path):
    """clip id -> gender ('male'|'female'|'unclear') from a dataset manifest."""
    with open(manifest_path) as f:
        m = json.load(f)
    return {cl["id"]: cl["gender"] for cl in m["clips"]}


def _read_clip_features(pq_path):
    """Return (75, 334) float32 array of per-frame pose features (f01_ column order).

    Unlike the exercise loader we do NOT append the parquet `sex` column: it would be a
    target leak (and is an unset placeholder anyway).

    These parquets are single-row and ultra-wide (25k cols), so we read once and slice via a
    single pass over the column list (the per-frame .startswith scan was the cost driver).
    """
    df = pd.read_parquet(pq_path)
    assert len(df) == 1, f"Expected 1 row, got {len(df)} in {pq_path}"
    row = df.iloc[0]
    # Single pass: bucket every f{NN}_ feature column by its frame index, in column order.
    frame_cols = [[] for _ in range(NUM_FRAMES)]
    for c in df.columns:
        if c[0] == "f" and c[1:3].isdigit() and c[3] == "_":
            fi = int(c[1:3])
            if 1 <= fi <= NUM_FRAMES:
                frame_cols[fi - 1].append(c)
    frame_features = []
    for fi in range(NUM_FRAMES):
        cols = frame_cols[fi]
        assert len(cols) == NUM_FEATURES_PER_FRAME, \
            f"Frame {fi+1} has {len(cols)} features, expected {NUM_FEATURES_PER_FRAME}"
        frame_features.append(row[cols].values.astype(np.float32))
    return np.stack(frame_features)  # (75, 334)


def load_all_data():
    """Load every clip across the three exercise sources, labeled by manifest gender.

    Returns:
        X:   tensor (N, 75, 334) float32
        y:   tensor (N,) int64  (0=male, 1=female, 2=unclear)
        ids: list of (exercise, clip_id)
    """
    X_list, y_list, ids = [], [], []
    skipped = 0
    for exercise, data_dir, pattern, manifest in SOURCES:
        gender_map = _load_gender_map(manifest)
        files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        for pq in files:
            cid = _clip_id_from_path(pq)
            g = gender_map.get(cid)
            if g not in GENDER_TO_IDX:
                skipped += 1
                continue
            X_list.append(_read_clip_features(pq))
            y_list.append(GENDER_TO_IDX[g])
            ids.append((exercise, cid))
    if skipped:
        print(f"[prepare_sex] skipped {skipped} parquet(s) with no/invalid manifest gender")
    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.long)
    return X, y, ids


def load_all_data_cached():
    """Cache raw load keyed on the full set of parquet filenames across the three dirs."""
    import hashlib
    all_files = []
    for _, data_dir, pattern, _ in SOURCES:
        all_files += sorted(glob.glob(os.path.join(data_dir, pattern)))
    key = hashlib.md5("|".join(os.path.basename(f) for f in all_files).encode()).hexdigest()[:12]
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f"sex_{len(all_files)}_{key}.pt")
    if os.path.exists(cp):
        b = torch.load(cp)
        return b["X"], b["y"], b["ids"]
    X, y, ids = load_all_data()
    torch.save({"X": X, "y": y, "ids": ids}, cp)
    return X, y, ids


def frame_feature_names():
    """Per-frame feature names (334) in the loader's column order (f01_ stripped)."""
    for _, data_dir, pattern, _ in SOURCES:
        files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if files:
            df = pd.read_parquet(files[0])
            return [c[4:] for c in df.columns if c.startswith("f01_")]
    raise RuntimeError("no parquet files found")


def group_frame_mask(feature_group="mp"):
    """Boolean mask over the 334 per-frame channels.

    feature_group: 'mp' (MediaPipe-only, default) | 'dp' (DensePose) | 'all'.
    No sex channel here (excluded as a target leak), so mask length == 334.
    """
    names = frame_feature_names()
    if feature_group == "mp":
        keep = [n.startswith(MP_GROUPS) for n in names]
    elif feature_group == "dp":
        keep = [n.startswith(DP_GROUPS) for n in names]
    else:
        keep = [True] * len(names)
    return torch.tensor(keep, dtype=torch.bool)


def aggregate(X):
    """(N, 75, C) -> (N, 7*C) via mean/std/min/max/median/q25/q75 across frames."""
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
    # by exercise
    from collections import Counter
    by_ex = Counter(ex for ex, _ in ids)
    print("\nBy exercise:", dict(by_ex))
    flat = X.reshape(-1, X.shape[-1])
    print(f"\nNaN: {torch.isnan(flat).sum().item()}  Inf: {torch.isinf(flat).sum().item()}")
    mask = group_frame_mask("mp")
    Xa = aggregate(X[:, :, mask])
    print(f"MediaPipe per-frame channels: {int(mask.sum())}  aggregated dim: {Xa.shape[1]}")
