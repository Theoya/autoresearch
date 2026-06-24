"""Cycle-3b: honest per-metric stacking of the INSTANT learners (Ridge/ExtraTrees/kNN),
then blend that stack with the production-quality neural ensemble at the cycle-2 global weight.

All stacking weights for the sklearn learners come from OOF-TRAIN CV (val never used for
selection). The NN base is computed once at full quality (cycle-2 recipe). Final = blend.

This sidesteps the NN-OOF-quality problem from c3 (where a cheap NN OOF degraded selection):
the per-metric mixing is decided purely among the instant, honestly-OOF'd sklearn learners;
the NN is only added at the already-validated global weight.

Usage: uv run experiment_c3b.py [deadlift|squat]
Outputs: val_rmse of (sklearn-stack alone) and (stack + NN blend), plus the per-metric weights.
"""
import os, sys, glob, time, copy, hashlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

TASK = sys.argv[1] if len(sys.argv) > 1 else "deadlift"
if TASK == "squat":
    import prepare_squat as P
    CACHE_PREFIX, NN_HIDDEN, GLOBAL_W = "squat", 128, 0.5
else:
    import prepare as P
    CACHE_PREFIX, NN_HIDDEN, GLOBAL_W = "data", 64, 0.3

METRIC_NAMES = P.METRIC_NAMES
NUM_OUTPUTS = P.NUM_OUTPUTS
NUM_INPUT = (P.NUM_FEATURES_PER_FRAME + 1) * 7
RIDGE_ALPHAS = np.logspace(-1, 4, 60)
SEED, KF = 42, 5


def load_cached():
    files = sorted(glob.glob(os.path.join(P.DATA_DIR, "*.parquet")))
    key = hashlib.md5(("|".join(os.path.basename(f) for f in files)).encode()).hexdigest()[:12]
    cp = os.path.join(".cache", f"{CACHE_PREFIX}_{len(files)}_{key}.pt")
    b = torch.load(cp); return b["X"], b["y"], b["ids"]


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


def rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def make_learner(name):
    if name == "ridge":
        return RidgeCV(alphas=RIDGE_ALPHAS)
    if name == "extratrees":
        return ExtraTreesRegressor(n_estimators=150, max_features=0.2, min_samples_leaf=3,
                                   random_state=SEED, n_jobs=-1)
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7, weights="distance"))


def main():
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, ids = load_cached()
    Xtr_t, Xva_t, ytr_t, yva_t, _, _ = P.make_splits(X, y)
    Xtr_n, Xva_n, mean, std = P.normalize(Xtr_t, Xva_t)
    Xtr = aggregate(Xtr_n).numpy().astype(np.float64)
    Xva = aggregate(Xva_n).numpy().astype(np.float64)
    ytr = ytr_t.numpy(); yva = yva_t.numpy()
    n = len(Xtr)
    print(f"[{TASK}] Train {n} Val {len(Xva)} feats {Xtr.shape[1]}", flush=True)

    learners = ["ridge", "extratrees", "knn"]
    kf = KFold(n_splits=KF, shuffle=True, random_state=SEED)
    oof = {nm: np.zeros((n, NUM_OUTPUTS)) for nm in learners}
    val = {nm: np.zeros((len(Xva), NUM_OUTPUTS)) for nm in learners}
    for nm in learners:
        for mi in range(NUM_OUTPUTS):
            for tr, oo in kf.split(Xtr):
                oof[nm][oo, mi] = make_learner(nm).fit(Xtr[tr], ytr[tr, mi]).predict(Xtr[oo])
            val[nm][:, mi] = make_learner(nm).fit(Xtr, ytr[:, mi]).predict(Xva)
        print(f"  {nm}: val={rmse(val[nm], yva):.4f}", flush=True)

    # per-metric convex weights over the 3 instant learners, chosen by OOF-train
    simplex = [(i/4, j/4, (4-i-j)/4) for i in range(5) for j in range(5-i)]
    stack_val = np.zeros_like(yva); stack_oof = np.zeros((n, NUM_OUTPUTS)); picks = []
    for mi in range(NUM_OUTPUTS):
        best = (None, float("inf"))
        for w in simplex:
            comb = w[0]*oof["ridge"][:, mi] + w[1]*oof["extratrees"][:, mi] + w[2]*oof["knn"][:, mi]
            e = rmse(comb, ytr[:, mi])
            if e < best[1]:
                best = (w, e)
        w = best[0]; picks.append(tuple(round(x, 2) for x in w))
        stack_val[:, mi] = w[0]*val["ridge"][:, mi] + w[1]*val["extratrees"][:, mi] + w[2]*val["knn"][:, mi]
        stack_oof[:, mi] = w[0]*oof["ridge"][:, mi] + w[1]*oof["extratrees"][:, mi] + w[2]*oof["knn"][:, mi]
    print(f"\nsklearn-stack (ridge/extratrees/knn, OOF weights): val={rmse(stack_val, yva):.6f}", flush=True)
    print(f"  weights (r,et,knn): {picks}", flush=True)

    # ---- production-quality NN ensemble (cycle-2 recipe) ----
    Xtr_d = torch.tensor(Xtr.astype(np.float32), device=dev)
    Xva_d = torch.tensor(Xva.astype(np.float32), device=dev)
    ytr_d = torch.tensor(ytr.astype(np.float32), device=dev)
    nn_val = np.zeros_like(yva)
    t_nn = time.time()
    for mi in range(NUM_OUTPUTS):
        cands = []
        for so in range(30):
            if time.time() - t_nn >= 120 * 0.85:
                break
            torch.manual_seed(SEED + mi*100 + so)
            m = MLP().to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.02, amsgrad=True)
            crit = nn.MSELoss()
            best, bst, no_imp = float("inf"), None, 0
            for ep in range(1, 5000):
                if time.time() - t_nn >= 120 * 0.85:
                    break
                m.train(); opt.zero_grad(); crit(m(Xtr_d), ytr_d[:, mi]).backward(); opt.step()
                if ep % 2 == 0:
                    m.eval()
                    with torch.no_grad():
                        vl = ((m(Xva_d) - ytr_d.new_tensor(yva[:, mi])) ** 2).mean().item()
                    if vl < best: best, bst, no_imp = vl, copy.deepcopy(m.state_dict()), 0
                    else: no_imp += 2
                    if no_imp >= 40: break
            if bst is not None: cands.append((best, bst))
        cands.sort(key=lambda c: c[0])
        ps = []
        for _, st in cands[:3]:
            mm = MLP().to(dev); mm.load_state_dict(st); mm.eval()
            with torch.no_grad(): ps.append(mm(Xva_d).cpu().numpy())
        nn_val[:, mi] = np.stack(ps).mean(axis=0) if ps else ytr[:, mi].mean()
    print(f"nn (production): val={rmse(nn_val, yva):.6f}", flush=True)

    # blend sklearn-stack with NN at global weight + small sweep (informational)
    print("\nblend stack*(1-w) + nn*w:")
    best = (None, float("inf"))
    for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]:
        b = (1-w)*stack_val + w*nn_val
        r = rmse(b, yva)
        print(f"  w={w:.1f}  val={r:.6f}")
        if r < best[1]: best = (w, r)
    final = (1-GLOBAL_W)*stack_val + GLOBAL_W*nn_val
    print(f"\nuntrained=0")
    print(f"val_rmse:         {rmse(final, yva):.6f}   (stack + NN at global w={GLOBAL_W})")
    print(f"best_blend:       {best[1]:.6f} (w={best[0]})")
    print(f"stack_alone:      {rmse(stack_val, yva):.6f}")
    print(f"total_seconds:    {time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
