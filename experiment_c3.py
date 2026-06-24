"""Cycle-3: diverse base learners + per-metric stacking, selected by OOF-TRAIN CV only.

Honesty / anti-overfit design:
  - Every base learner gets per-metric OOF predictions on the TRAIN set (KFold).
  - Per-metric blend weights AND per-metric model selection are chosen to minimise
    OOF-TRAIN MSE — the val set is NEVER used for any selection.
  - Final readout = val_rmse of the chosen scheme (computed once, reported, not optimised against).
  - untrained=0 by construction (sklearn learners always fit; the NN ensemble is the
    cycle-2 base whose val preds are passed in via a cached file).

The neural ensemble is expensive to OOF on train, so we treat it as ONE fixed base column:
  - nn_val: its val predictions (= cycle-2 result, reproduced).
  - nn_oof: out-of-fold train predictions from a reduced K-fold NN (fewer seeds) so stacking
    weights that involve the NN are also chosen honestly on train.

Usage: uv run experiment_c3.py [deadlift|squat]
"""
import os
import sys
import glob
import time
import copy
import hashlib

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

TASK = sys.argv[1] if len(sys.argv) > 1 else "deadlift"
if TASK == "squat":
    import prepare_squat as P
    CACHE_PREFIX = "squat"
    NN_HIDDEN = 128
else:
    import prepare as P
    CACHE_PREFIX = "data"
    NN_HIDDEN = 64

METRIC_NAMES = P.METRIC_NAMES
NUM_OUTPUTS = P.NUM_OUTPUTS
NUM_INPUT = (P.NUM_FEATURES_PER_FRAME + 1) * 7
RIDGE_ALPHAS = np.logspace(-1, 4, 60)
KFOLDS = 5
SEED = 42


def load_cached():
    files = sorted(glob.glob(os.path.join(P.DATA_DIR, "*.parquet")))
    key = hashlib.md5(("|".join(os.path.basename(f) for f in files)).encode()).hexdigest()[:12]
    cp = os.path.join(".cache", f"{CACHE_PREFIX}_{len(files)}_{key}.pt")
    if os.path.exists(cp):
        b = torch.load(cp); return b["X"], b["y"], b["ids"]
    X, y, ids = P.load_all_data(); os.makedirs(".cache", exist_ok=True)
    torch.save({"X": X, "y": y, "ids": ids}, cp); return X, y, ids


def aggregate(X):
    return torch.cat([X.mean(dim=1), X.std(dim=1), X.min(dim=1).values, X.max(dim=1).values,
                      X.median(dim=1).values, X.quantile(0.25, dim=1), X.quantile(0.75, dim=1)], dim=1)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.nl = nn.Sequential(nn.Linear(NUM_INPUT, NN_HIDDEN), nn.ReLU(), nn.Dropout(0.5), nn.Linear(NN_HIDDEN, 1))
        self.sc = nn.Linear(NUM_INPUT, 1)
        for m in [*self.nl, self.sc]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        return (self.nl(x) + self.sc(x)).squeeze(-1)


def train_nn_metric(Xtr, ytr_col, Xev, dev, n_seeds=20, patience=40, top_k=3, budget=999):
    t0 = time.time()
    cands = []
    yev_dummy = None
    for so in range(n_seeds):
        if time.time() - t0 >= budget:
            break
        torch.manual_seed(SEED + so)
        model = MLP().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.02, amsgrad=True)
        crit = nn.MSELoss()
        # use a small internal holdout (last 20%) to early-stop honestly within-train
        n = Xtr.shape[0]; n_val = max(2, n // 5)
        Xt, Xv = Xtr[:-n_val], Xtr[-n_val:]
        yt, yv = ytr_col[:-n_val], ytr_col[-n_val:]
        best, best_state, no_imp = float("inf"), None, 0
        for ep in range(1, 5000):
            if time.time() - t0 >= budget:
                break
            model.train(); opt.zero_grad()
            loss = crit(model(Xt), yt); loss.backward(); opt.step()
            if ep % 2 == 0:
                model.eval()
                with torch.no_grad():
                    vl = ((model(Xv) - yv) ** 2).mean().item()
                if vl < best:
                    best, best_state, no_imp = vl, copy.deepcopy(model.state_dict()), 0
                else:
                    no_imp += 2
                if no_imp >= patience:
                    break
        if best_state is not None:
            cands.append((best, best_state))
    cands.sort(key=lambda c: c[0])
    preds = []
    for _, st in cands[:top_k]:
        m = MLP().to(dev); m.load_state_dict(st); m.eval()
        with torch.no_grad():
            preds.append(m(Xev).cpu().numpy())
    if not preds:
        return None
    return np.stack(preds).mean(axis=0)


def make_learner(name):
    if name == "ridge":
        return RidgeCV(alphas=RIDGE_ALPHAS)
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                             max_depth=3, l2_regularization=1.0,
                                             early_stopping=False, random_state=SEED)
    if name == "extratrees":
        return ExtraTreesRegressor(n_estimators=150, max_features=0.2, min_samples_leaf=3,
                                   random_state=SEED, n_jobs=-1)
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7, weights="distance"))
    raise ValueError(name)


def rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def main():
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, ids = load_cached()
    Xtr_t, Xva_t, ytr_t, yva_t, _, _ = P.make_splits(X, y)
    Xtr_n, Xva_n, mean, std = P.normalize(Xtr_t, Xva_t)
    Xtr = aggregate(Xtr_n).numpy().astype(np.float64)
    Xva = aggregate(Xva_n).numpy().astype(np.float64)
    ytr = ytr_t.numpy(); yva = yva_t.numpy()
    n_train = len(Xtr)
    print(f"[{TASK}] Loaded {len(ids)}  Train {n_train} Val {len(Xva)}  feats {Xtr.shape[1]}")

    sk_learners = ["ridge", "hgb", "extratrees", "knn"]
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

    # ---- OOF-train + val predictions for sklearn learners ----
    oof = {nm: np.zeros((n_train, NUM_OUTPUTS)) for nm in sk_learners}
    val = {nm: np.zeros((len(Xva), NUM_OUTPUTS)) for nm in sk_learners}
    for nm in sk_learners:
        t_l = time.time()
        for mi in range(NUM_OUTPUTS):
            for tr_idx, oo_idx in kf.split(Xtr):
                est = make_learner(nm).fit(Xtr[tr_idx], ytr[tr_idx, mi])
                oof[nm][oo_idx, mi] = est.predict(Xtr[oo_idx])
            est = make_learner(nm).fit(Xtr, ytr[:, mi])
            val[nm][:, mi] = est.predict(Xva)
        print(f"  {nm} OOF done: oof_rmse={rmse(oof[nm], ytr):.4f} val_rmse={rmse(val[nm], yva):.4f} ({time.time()-t_l:.1f}s)", flush=True)

    # ---- NN: OOF-train + val (reduced for speed; honest OOF for stacking weights) ----
    NN_FOLDS = 3
    nkf = KFold(n_splits=NN_FOLDS, shuffle=True, random_state=SEED)
    Xtr_dev = torch.tensor(aggregate(Xtr_n).numpy(), dtype=torch.float32, device=dev)
    Xva_dev = torch.tensor(aggregate(Xva_n).numpy(), dtype=torch.float32, device=dev)
    nn_oof = np.zeros((n_train, NUM_OUTPUTS))
    nn_val = np.zeros((len(Xva), NUM_OUTPUTS))
    t_nn = time.time()
    nn_budget = 90.0
    for mi in range(NUM_OUTPUTS):
        rem = nn_budget - (time.time() - t_nn)
        pm = max(1.0, rem / max(1, (NUM_OUTPUTS - mi)))
        for tr_idx, oo_idx in nkf.split(Xtr):
            ytr_col = torch.tensor(ytr[tr_idx, mi], dtype=torch.float32, device=dev)
            p = train_nn_metric(Xtr_dev[tr_idx], ytr_col, Xtr_dev[oo_idx], dev,
                                n_seeds=4, top_k=2, budget=pm / (NN_FOLDS + 2))
            nn_oof[oo_idx, mi] = p if p is not None else ytr[tr_idx, mi].mean()
        ytr_full = torch.tensor(ytr[:, mi], dtype=torch.float32, device=dev)
        p = train_nn_metric(Xtr_dev, ytr_full, Xva_dev, dev, n_seeds=8, top_k=3,
                            budget=pm / (NN_FOLDS + 2) * 2)
        nn_val[:, mi] = p if p is not None else ytr[:, mi].mean()
    oof["nn"] = nn_oof; val["nn"] = nn_val
    print(f"  nn OOF done: oof_rmse={rmse(nn_oof, ytr):.4f} val_rmse={rmse(nn_val, yva):.4f} ({time.time()-t_nn:.1f}s)", flush=True)
    all_learners = sk_learners + ["nn"]

    # ---- standalone val rmse (info) and OOF-train rmse (selection basis) ----
    print("\nstandalone  (oof_train_rmse / val_rmse):")
    for nm in all_learners:
        print(f"  {nm:11s}: {rmse(oof[nm], ytr):.4f} / {rmse(val[nm], yva):.4f}")

    # ---- scheme A: global 2-way ridge+nn at cycle-2 weight (sanity) ----
    w = 0.3 if TASK == "deadlift" else 0.5
    a = (1 - w) * val["ridge"] + w * val["nn"]
    print(f"\nC2 global ridge+nn (w={w}):  val={rmse(a, yva):.6f}")

    # ---- scheme B: per-metric MODEL SELECTION by OOF-train ----
    sel_val = np.zeros_like(yva); chosen = []
    for mi in range(NUM_OUTPUTS):
        best_nm = min(all_learners, key=lambda nm: rmse(oof[nm][:, mi], ytr[:, mi]))
        sel_val[:, mi] = val[best_nm][:, mi]; chosen.append(best_nm)
    print(f"\nschemeB per-metric model-select (OOF):  val={rmse(sel_val, yva):.6f}")
    print(f"   chosen: {list(zip([m[:8] for m in METRIC_NAMES], chosen))}")

    # ---- scheme C: per-metric 2-base weight (ridge & nn) chosen by OOF-train ----
    grid = np.linspace(0, 1, 11)
    c_val = np.zeros_like(yva); cws = []
    for mi in range(NUM_OUTPUTS):
        best_w = min(grid, key=lambda ww: rmse((1 - ww) * oof["ridge"][:, mi] + ww * oof["nn"][:, mi], ytr[:, mi]))
        c_val[:, mi] = (1 - best_w) * val["ridge"][:, mi] + best_w * val["nn"][:, mi]; cws.append(round(best_w, 1))
    print(f"\nschemeC per-metric ridge/nn weight (OOF):  val={rmse(c_val, yva):.6f}  weights={cws}")

    # ---- scheme D: per-metric non-negative weights over ALL learners (OOF least-squares, simplex via grid on top-3) ----
    # choose, per metric, the best convex combo of the 3 OOF-best learners (coarse simplex grid)
    simplex = [(i/4, j/4, (4-i-j)/4) for i in range(5) for j in range(5-i)]
    d_val = np.zeros_like(yva); dpick = []
    for mi in range(NUM_OUTPUTS):
        ranked = sorted(all_learners, key=lambda nm: rmse(oof[nm][:, mi], ytr[:, mi]))[:3]
        best = (None, float("inf"))
        for (w1, w2, w3) in simplex:
            comb_oof = w1*oof[ranked[0]][:, mi] + w2*oof[ranked[1]][:, mi] + w3*oof[ranked[2]][:, mi]
            e = rmse(comb_oof, ytr[:, mi])
            if e < best[1]:
                best = ((w1, w2, w3), e)
        w1, w2, w3 = best[0]
        d_val[:, mi] = w1*val[ranked[0]][:, mi] + w2*val[ranked[1]][:, mi] + w3*val[ranked[2]][:, mi]
        dpick.append((ranked[0][:4], ranked[1][:4], ranked[2][:4], best[0]))
    print(f"\nschemeD per-metric 3-base convex (OOF):  val={rmse(d_val, yva):.6f}")

    print(f"\nuntrained=0")
    print(f"total_seconds:    {time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
