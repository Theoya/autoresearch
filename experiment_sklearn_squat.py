"""Cycle-2 experiment: per-metric sklearn linear models with per-metric alpha search.

Reuses the exact prepare.py data pipeline (load/split/normalize) and the same 7-stat
temporal aggregation as the D11 neural baseline, then fits per-metric Ridge / Lasso /
ElasticNet with cross-validated alpha. sklearn models train ~instantly so ALL 11 metrics
fit honestly within budget (no metric-starvation / mean-fallback possible).

Evaluated on the SAME fixed train/val split as train.py (prepare.make_splits, val_ratio=0.2),
with prepare.evaluate_rmse — directly comparable to D11's 1.609103.

Usage: uv run experiment_sklearn.py [model]   model in {ridge,lasso,elasticnet,all}
"""
import os
import sys
import glob
import time
import hashlib

import numpy as np
import torch

from prepare_squat import (
    NUM_FRAMES, NUM_FEATURES_PER_FRAME, NUM_OUTPUTS, METRIC_NAMES, DATA_DIR,
    load_all_data, make_splits, normalize, evaluate_rmse,
)
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline


def load_all_data_cached():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    key = hashlib.md5(("|".join(os.path.basename(f) for f in files)).encode()).hexdigest()[:12]
    cache_dir = ".cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"squat_{len(files)}_{key}.pt")
    if os.path.exists(cache_path):
        blob = torch.load(cache_path)
        return blob["X"], blob["y"], blob["ids"]
    X, y, ids = load_all_data()
    torch.save({"X": X, "y": y, "ids": ids}, cache_path)
    return X, y, ids


def aggregate(X, nine=False):
    """(N,75,335) -> 7-stat (2345) or 9-stat (3015)."""
    parts = [X.mean(dim=1), X.std(dim=1), X.min(dim=1).values, X.max(dim=1).values,
             X.median(dim=1).values, X.quantile(0.25, dim=1), X.quantile(0.75, dim=1)]
    if nine:
        parts += [X.quantile(0.10, dim=1), X.quantile(0.90, dim=1)]
    return torch.cat(parts, dim=1)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()

    nine = "9stat" in sys.argv
    X, y, ids = load_all_data_cached()
    print(f"Loaded {len(ids)} samples")
    X_train, X_val, y_train, y_val, tr_idx, va_idx = make_splits(X, y)
    X_train, X_val, mean, std = normalize(X_train, X_val)
    Xtr = aggregate(X_train, nine).numpy()
    Xva = aggregate(X_val, nine).numpy()
    ytr = y_train.numpy()
    yva = y_val.numpy()
    print(f"Train: {len(Xtr)}, Val: {len(Xva)}, features: {Xtr.shape[1]}  (9stat={nine})")

    n_train = len(Xtr)
    alphas = np.logspace(-2, 4, 25)
    alphas_fine = np.logspace(-1, 4, 60)
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

    def build(model_name):
        if model_name == "ridge":
            return RidgeCV(alphas=alphas, cv=min(5, n_train))
        if model_name == "ridge_loo":
            return RidgeCV(alphas=alphas_fine)
        if model_name == "ridge_fine":
            return RidgeCV(alphas=alphas_fine, cv=min(5, n_train))
        if model_name == "lasso":
            return LassoCV(alphas=alphas, cv=min(5, n_train), max_iter=20000, n_jobs=-1)
        if model_name == "elasticnet":
            return ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=min(5, n_train),
                                max_iter=20000, n_jobs=-1)
        if model_name.startswith("ridge_fs"):
            k = int(model_name.split("fs")[1])
            return make_pipeline(SelectKBest(f_regression, k=min(k, Xtr.shape[1])),
                                 RidgeCV(alphas=alphas_fine))
        if model_name.startswith("ridge_pca"):
            k = int(model_name.split("pca")[1])
            return make_pipeline(PCA(n_components=min(k, n_train - 1)),
                                 RidgeCV(alphas=alphas_fine))
        raise ValueError(model_name)

    presets = {
        "all": ["ridge", "lasso", "elasticnet"],
        "ridges": ["ridge", "ridge_loo", "ridge_fine"],
        "fs": ["ridge_fs100", "ridge_fs300", "ridge_fs600", "ridge_fs1000"],
        "pca": ["ridge_pca20", "ridge_pca40", "ridge_pca80", "ridge_pca120"],
    }
    model_list = presets.get(which, [which])

    for model_name in model_list:
        t_m = time.time()
        preds = np.zeros_like(yva)
        untrained = 0
        chosen_alphas = []
        for mi, name in enumerate(METRIC_NAMES):
            try:
                est = build(model_name)
                est.fit(Xtr, ytr[:, mi])
                preds[:, mi] = est.predict(Xva)
                chosen_alphas.append(getattr(est, "alpha_", float("nan")))
            except Exception as e:
                # honest fallback would be mean — count as untrained
                preds[:, mi] = ytr[:, mi].mean()
                untrained += 1
                print(f"  {name}: FAILED {e}")
        val_rmse = evaluate_rmse(torch.tensor(preds), torch.tensor(yva))
        print(f"\n=== {model_name} ===")
        print(f"untrained={untrained}")
        print(f"val_rmse:         {val_rmse:.6f}")
        print(f"  median alpha: {np.nanmedian(chosen_alphas):.3f}  time: {time.time()-t_m:.1f}s")
        # per-metric
        for mi, name in enumerate(METRIC_NAMES):
            rmse = float(np.sqrt(((preds[:, mi] - yva[:, mi]) ** 2).mean()))
            print(f"    {name:22s}: {rmse:.4f}")

    print(f"\ntotal_seconds:    {time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
