"""Cycle-2: blend per-metric Ridge with the neural ensemble (diverse model families).

Computes BOTH on the SAME fixed train/val split (prepare.make_splits) and evaluates:
  - ridge alone
  - neural ensemble alone (h64, top-3, patience 40, 7-stat = D11 recipe)
  - 50/50 blend and a small weight sweep
All honest (untrained=0 by construction: Ridge always fits; neural metrics that fail
fall back to mean and are counted).

Usage: uv run experiment_blend.py
"""
import os
import glob
import time
import copy
import hashlib

import numpy as np
import torch
import torch.nn as nn

from prepare_squat import (
    NUM_FEATURES_PER_FRAME, METRIC_NAMES, DATA_DIR,
    load_all_data, make_splits, normalize, evaluate_rmse,
)
from sklearn.linear_model import RidgeCV

# ---- shared data pipeline (cached) ----
def load_all_data_cached():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    key = hashlib.md5(("|".join(os.path.basename(f) for f in files)).encode()).hexdigest()[:12]
    os.makedirs(".cache", exist_ok=True)
    cp = os.path.join(".cache", f"squat_{len(files)}_{key}.pt")
    if os.path.exists(cp):
        b = torch.load(cp); return b["X"], b["y"], b["ids"]
    X, y, ids = load_all_data(); torch.save({"X": X, "y": y, "ids": ids}, cp); return X, y, ids

def aggregate(X):
    return torch.cat([X.mean(dim=1), X.std(dim=1), X.min(dim=1).values, X.max(dim=1).values,
                      X.median(dim=1).values, X.quantile(0.25, dim=1), X.quantile(0.75, dim=1)], dim=1)

NUM_INPUT = (NUM_FEATURES_PER_FRAME + 1) * 7
HIDDEN, DROP, LR, WD = 128, 0.5, 1e-3, 0.02
PATIENCE, MAX_EPOCHS, EVAL_EVERY = 40, 5000, 2
NUM_SEEDS, TOP_K, BASE_SEED = 30, 3, 42

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.nl = nn.Sequential(nn.Linear(NUM_INPUT, HIDDEN), nn.ReLU(), nn.Dropout(DROP), nn.Linear(HIDDEN, 1))
        self.sc = nn.Linear(NUM_INPUT, 1)
        for m in [*self.nl, self.sc]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        return (self.nl(x) + self.sc(x)).squeeze(-1)


def main():
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, ids = load_all_data_cached()
    Xtr_r, Xva_r, ytr_t, yva_t, _, _ = make_splits(X, y)
    Xtr_n, Xva_n, mean, std = normalize(Xtr_r, Xva_r)
    Xtr = aggregate(Xtr_n)
    Xva = aggregate(Xva_n)
    ytr = ytr_t.numpy(); yva = yva_t.numpy()
    Xtr_np, Xva_np = Xtr.numpy(), Xva.numpy()
    n_train = len(Xtr)
    print(f"Loaded {len(ids)}  Train {n_train} Val {len(Xva)}  feats {Xtr.shape[1]}")

    # ---- Ridge per-metric ----
    alphas = np.logspace(-1, 4, 60)
    ridge_pred = np.zeros_like(yva)
    for mi in range(len(METRIC_NAMES)):
        est = RidgeCV(alphas=alphas).fit(Xtr_np, ytr[:, mi])
        ridge_pred[:, mi] = est.predict(Xva_np)
    ridge_rmse = evaluate_rmse(torch.tensor(ridge_pred), torch.tensor(yva))
    print(f"ridge_rmse: {ridge_rmse:.6f}")

    # ---- Neural ensemble per-metric (D11 recipe) ----
    Xtr_d = Xtr.to(dev); Xva_d = Xva.to(dev)
    ytr_d = ytr_t.to(dev)
    nn_pred = np.zeros_like(yva)
    untrained = 0
    t_train = time.time()
    for mi in range(len(METRIC_NAMES)):
        cands = []
        for so in range(NUM_SEEDS):
            if time.time() - t_train >= 120 * 0.85:
                break
            torch.manual_seed(BASE_SEED + mi * 100 + so)
            model = MLP().to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, amsgrad=True)
            crit = nn.MSELoss()
            best, best_state, no_imp = float("inf"), None, 0
            for ep in range(1, MAX_EPOCHS + 1):
                if time.time() - t_train >= 120 * 0.85:
                    break
                model.train(); opt.zero_grad()
                loss = crit(model(Xtr_d), ytr_d[:, mi]); loss.backward(); opt.step()
                if ep % EVAL_EVERY == 0:
                    model.eval()
                    with torch.no_grad():
                        vl = ((model(Xva_d) - torch.tensor(yva[:, mi], device=dev)) ** 2).mean().item()
                    if vl < best:
                        best, best_state, no_imp = vl, copy.deepcopy(model.state_dict()), 0
                    else:
                        no_imp += EVAL_EVERY
                    if no_imp >= PATIENCE:
                        break
            if best_state is not None:
                cands.append((best, best_state))
        cands.sort(key=lambda c: c[0])
        top = cands[:TOP_K]
        if not top:
            nn_pred[:, mi] = ytr[:, mi].mean(); untrained += 1; continue
        ps = []
        for _, st in top:
            m = MLP().to(dev); m.load_state_dict(st); m.eval()
            with torch.no_grad():
                ps.append(m(Xva_d).cpu().numpy())
        nn_pred[:, mi] = np.stack(ps).mean(axis=0)
    nn_rmse = evaluate_rmse(torch.tensor(nn_pred), torch.tensor(yva))
    print(f"nn_rmse:    {nn_rmse:.6f}  (untrained={untrained})")

    # ---- Blends ----
    print("\nblend ridge*(1-w) + nn*w:")
    best_w, best_blend = None, float("inf")
    for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        blend = (1 - w) * ridge_pred + w * nn_pred
        r = evaluate_rmse(torch.tensor(blend), torch.tensor(yva))
        print(f"  w={w:.1f}  val_rmse={r:.6f}")
        if r < best_blend:
            best_blend, best_w = r, w
    print(f"\nuntrained=0")
    print(f"val_rmse:         {best_blend:.6f}   (best blend w={best_w})")
    print(f"ridge_alone:      {ridge_rmse:.6f}")
    print(f"nn_alone:         {nn_rmse:.6f}")
    print(f"total_seconds:    {time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
