"""
Data prep for DEADLIFT form regression on the RE-LABELED (grounded-rubric) clips ONLY.

Isolates the re-label effect: loads only the claude re-labeled deadlift parquets
(consistent full-range labels), excluding the 78 original-convention clips that
contaminate the mixed kinetics/labelled dir. 11 deadlift metrics.

Usage: python prepare_dl_relabel.py
"""
import os, json, glob
import numpy as np
import pandas as pd
import torch

NUM_FRAMES = 75
NUM_FEATURES_PER_FRAME = 334
NUM_OUTPUTS = 11
DATA_DIR = "T:/clipforge/datasets/kinetics/labelled"
GLOB = "claude_*.parquet"  # re-labeled claude clips only (excludes the 78 originals)
TIME_BUDGET = 12000

METRIC_NAMES = [
    "leftFootPosition", "rightFootPosition", "grip", "barPath", "hipPosition",
    "spineNeutrality", "shoulderPosition", "coreBracing", "extensionTiming",
    "lockoutPosition", "controlledDescent",
]


def load_all_data():
    parquet_files = sorted(glob.glob(os.path.join(DATA_DIR, GLOB)))
    assert len(parquet_files) > 0, f"No parquet files found in {DATA_DIR}"
    X_list, y_list, ids = [], [], []
    for pq_path in parquet_files:
        sample_id = os.path.splitext(os.path.basename(pq_path))[0]
        json_path = pq_path.replace(".parquet", ".json")
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            metrics = json.load(f)["metrics"]
        if not all(k in metrics for k in METRIC_NAMES):
            continue
        y_vec = [metrics[name] for name in METRIC_NAMES]
        df = pd.read_parquet(pq_path)
        assert len(df) == 1
        row = df.iloc[0]
        sex_val = float(row["sex"])
        frame_features = []
        for frame_idx in range(1, NUM_FRAMES + 1):
            prefix = f"f{frame_idx:02d}_"
            frame_cols = [c for c in df.columns if c.startswith(prefix)]
            assert len(frame_cols) == NUM_FEATURES_PER_FRAME
            vals = row[frame_cols].values.astype(np.float32)
            vals = np.append(vals, sex_val)
            frame_features.append(vals)
        X_list.append(np.stack(frame_features))
        y_list.append(y_vec)
        ids.append(sample_id)
    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)
    return X, y, ids


def make_splits(X, y, val_ratio=0.15, seed=42):
    N = X.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_val = max(1, int(N * val_ratio))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx], train_idx, val_idx


def normalize(X_train, X_val):
    mean = X_train.mean(dim=(0, 1), keepdim=True)
    std = X_train.std(dim=(0, 1), keepdim=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, mean, std


def evaluate_rmse(predictions, targets):
    return (((predictions - targets) ** 2).mean()).sqrt().item()


if __name__ == "__main__":
    X, y, ids = load_all_data()
    print(f"Loaded {len(ids)} samples")
    Xtr, Xva, ytr, yva, _, _ = make_splits(X, y)
    base = evaluate_rmse(ytr.mean(dim=0, keepdim=True).expand_as(yva), yva)
    print(f"Mean-prediction baseline RMSE: {base:.6f}")
