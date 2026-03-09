"""
Data preparation and evaluation for deadlift form regression.

Loads pose/DensePose features from parquet files and form scores from JSON labels.
Each sample = 75 frames × 334 features + sex scalar → 11 form metric scores (0-9).

Usage:
    python prepare.py   # load data, print stats
"""

import os
import json
import glob

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

NUM_FRAMES = 75
NUM_FEATURES_PER_FRAME = 334
NUM_OUTPUTS = 11
DATA_DIR = "T:/clipforge/datasets/kinetics/labelled"
TIME_BUDGET = 120  # training time budget in seconds

METRIC_NAMES = [
    "leftFootPosition", "rightFootPosition", "grip", "barPath",
    "hipPosition", "spineNeutrality", "shoulderPosition",
    "coreBracing", "extensionTiming", "lockoutPosition", "controlledDescent",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data():
    """
    Load all labelled samples from DATA_DIR.

    Returns:
        X: tensor (N, 75, 335) — 334 pose features + sex per frame
        y: tensor (N, 11) — form metric scores
        ids: list of sample ID strings
    """
    parquet_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    assert len(parquet_files) > 0, f"No parquet files found in {DATA_DIR}"

    X_list = []
    y_list = []
    ids = []

    for pq_path in parquet_files:
        sample_id = os.path.splitext(os.path.basename(pq_path))[0]
        json_path = pq_path.replace(".parquet", ".json")

        if not os.path.exists(json_path):
            print(f"  Skipping {sample_id}: no matching JSON")
            continue

        # Load labels
        with open(json_path) as f:
            label_data = json.load(f)
        metrics = label_data["metrics"]
        y_vec = [metrics[name] for name in METRIC_NAMES]

        # Load features
        df = pd.read_parquet(pq_path)
        assert len(df) == 1, f"Expected 1 row, got {len(df)} in {pq_path}"

        row = df.iloc[0]
        sex_val = float(row["sex"])

        # Extract frame features: columns are f01_*, f02_*, ..., f75_*
        frame_features = []
        for frame_idx in range(1, NUM_FRAMES + 1):
            prefix = f"f{frame_idx:02d}_"
            frame_cols = [c for c in df.columns if c.startswith(prefix)]
            assert len(frame_cols) == NUM_FEATURES_PER_FRAME, \
                f"Frame {frame_idx} has {len(frame_cols)} features, expected {NUM_FEATURES_PER_FRAME}"
            vals = row[frame_cols].values.astype(np.float32)
            # Append sex to each frame
            vals = np.append(vals, sex_val)
            frame_features.append(vals)

        X_sample = np.stack(frame_features)  # (75, 335)
        X_list.append(X_sample)
        y_list.append(y_vec)
        ids.append(sample_id)

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)  # (N, 75, 335)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)  # (N, 11)

    return X, y, ids


def make_splits(X, y, val_ratio=0.2, seed=42):
    """
    Split data into train/val sets.

    Returns:
        X_train, X_val, y_train, y_val, train_idx, val_idx
    """
    N = X.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_val = max(1, int(N * val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train = X[train_idx]
    X_val = X[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    return X_train, X_val, y_train, y_val, train_idx, val_idx


def normalize(X_train, X_val):
    """
    Per-feature z-score normalization across time and samples.
    Computed on train set, applied to both.

    Args:
        X_train: (N_train, 75, 335)
        X_val: (N_val, 75, 335)

    Returns:
        X_train_norm, X_val_norm, mean, std
    """
    # Compute stats over (samples, frames) → per-feature
    mean = X_train.mean(dim=(0, 1), keepdim=True)  # (1, 1, 335)
    std = X_train.std(dim=(0, 1), keepdim=True) + 1e-8  # (1, 1, 335)

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    return X_train_norm, X_val_norm, mean, std


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_rmse(predictions, targets):
    """
    Root Mean Squared Error across all 11 outputs jointly.

    Args:
        predictions: tensor (N, 11)
        targets: tensor (N, 11)

    Returns:
        float: RMSE (lower is better)
    """
    mse = ((predictions - targets) ** 2).mean()
    return mse.sqrt().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Data directory: {DATA_DIR}")
    print()

    X, y, ids = load_all_data()
    print(f"Loaded {len(ids)} samples")
    print(f"X shape: {X.shape}  (samples, frames, features)")
    print(f"y shape: {y.shape}  (samples, metrics)")
    print()

    # Label statistics
    print("Label statistics (per metric):")
    for i, name in enumerate(METRIC_NAMES):
        vals = y[:, i]
        print(f"  {name:25s}: mean={vals.mean():.2f}  std={vals.std():.2f}  "
              f"min={vals.min():.0f}  max={vals.max():.0f}")
    print()

    # Feature statistics
    print("Feature statistics (across all samples and frames):")
    flat = X.reshape(-1, X.shape[-1])
    n_nan = torch.isnan(flat).sum().item()
    n_inf = torch.isinf(flat).sum().item()
    print(f"  NaN count: {n_nan}")
    print(f"  Inf count: {n_inf}")
    print(f"  Feature range: [{flat.min():.4f}, {flat.max():.4f}]")
    print()

    # Train/val split
    X_train, X_val, y_train, y_val, train_idx, val_idx = make_splits(X, y)
    print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")

    # Dummy prediction (mean of training targets)
    y_mean = y_train.mean(dim=0, keepdim=True).expand_as(y_val)
    baseline_rmse = evaluate_rmse(y_mean, y_val)
    print(f"Mean-prediction baseline RMSE: {baseline_rmse:.6f}")
    print()
    print("Done! Ready to train.")
