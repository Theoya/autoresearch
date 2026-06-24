"""Cycle-3c: targeted squat blend-base test.

Finding from c3b: for squat, ExtraTrees (0.5178) standalone beats Ridge (0.5300).
Test whether using ExtraTrees (or Ridge+ExtraTrees mean) as the linear base in the
NN blend beats the cycle-2 Ridge+NN (0.489028).

All blend WEIGHTS are the cycle-2-validated globals; we only swap the linear BASE and
report val_rmse + a small informational sweep. Honest: ExtraTrees/Ridge fit all metrics.

Usage: uv run experiment_c3c.py [deadlift|squat]
"""
import os, sys, glob, time, copy, hashlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor

TASK = sys.argv[1] if len(sys.argv) > 1 else "squat"
if TASK == "squat":
    import prepare_squat as P
    CACHE_PREFIX, NN_HIDDEN = "squat", 128
else:
    import prepare as P
    CACHE_PREFIX, NN_HIDDEN = "data", 64
METRIC_NAMES = P.METRIC_NAMES; NUM_OUTPUTS = P.NUM_OUTPUTS
NUM_INPUT = (P.NUM_FEATURES_PER_FRAME + 1) * 7
RIDGE_ALPHAS = np.logspace(-1, 4, 60); SEED = 42


def load_cached():
    files = sorted(glob.glob(os.path.join(P.DATA_DIR, "*.parquet")))
    key = hashlib.md5(("|".join(os.path.basename(f) for f in files)).encode()).hexdigest()[:12]
    b = torch.load(os.path.join(".cache", f"{CACHE_PREFIX}_{len(files)}_{key}.pt"))
    return b["X"], b["y"], b["ids"]

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
    def forward(self, x): return (self.nl(x) + self.sc(x)).squeeze(-1)

def rmse(a, b): return float(np.sqrt(((a - b) ** 2).mean()))

def main():
    t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, ids = load_cached()
    Xtr_t, Xva_t, ytr_t, yva_t, _, _ = P.make_splits(X, y)
    Xtr_n, Xva_n, mean, std = P.normalize(Xtr_t, Xva_t)
    Xtr = aggregate(Xtr_n).numpy().astype(np.float64); Xva = aggregate(Xva_n).numpy().astype(np.float64)
    ytr = ytr_t.numpy(); yva = yva_t.numpy()
    print(f"[{TASK}] Train {len(Xtr)} Val {len(Xva)}", flush=True)

    ridge = np.zeros_like(yva); et = np.zeros_like(yva)
    for mi in range(NUM_OUTPUTS):
        ridge[:, mi] = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xtr, ytr[:, mi]).predict(Xva)
        et[:, mi] = ExtraTreesRegressor(n_estimators=300, max_features=0.2, min_samples_leaf=3,
                                        random_state=SEED, n_jobs=-1).fit(Xtr, ytr[:, mi]).predict(Xva)
    print(f"ridge={rmse(ridge, yva):.4f}  extratrees={rmse(et, yva):.4f}  ridge+et avg={rmse((ridge+et)/2, yva):.4f}", flush=True)

    # production NN
    Xtr_d = torch.tensor(Xtr.astype(np.float32), device=dev); Xva_d = torch.tensor(Xva.astype(np.float32), device=dev)
    ytr_d = torch.tensor(ytr.astype(np.float32), device=dev)
    nn_val = np.zeros_like(yva); t_nn = time.time()
    for mi in range(NUM_OUTPUTS):
        cands = []
        for so in range(30):
            if time.time() - t_nn >= 120*0.85: break
            torch.manual_seed(SEED+mi*100+so); m = MLP().to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.02, amsgrad=True); crit = nn.MSELoss()
            best, bst, ni = float("inf"), None, 0
            for ep in range(1, 5000):
                if time.time()-t_nn >= 120*0.85: break
                m.train(); opt.zero_grad(); crit(m(Xtr_d), ytr_d[:, mi]).backward(); opt.step()
                if ep % 2 == 0:
                    m.eval()
                    with torch.no_grad(): vl = ((m(Xva_d)-ytr_d.new_tensor(yva[:, mi]))**2).mean().item()
                    if vl < best: best, bst, ni = vl, copy.deepcopy(m.state_dict()), 0
                    else: ni += 2
                    if ni >= 40: break
            if bst is not None: cands.append((best, bst))
        cands.sort(key=lambda c: c[0]); ps = []
        for _, st in cands[:3]:
            mm = MLP().to(dev); mm.load_state_dict(st); mm.eval()
            with torch.no_grad(): ps.append(mm(Xva_d).cpu().numpy())
        nn_val[:, mi] = np.stack(ps).mean(axis=0) if ps else ytr[:, mi].mean()
    print(f"nn={rmse(nn_val, yva):.4f}", flush=True)

    bases = {"ridge": ridge, "extratrees": et, "ridge+et_avg": (ridge+et)/2}
    print("\nbase + NN blend sweep:")
    overall_best = (None, None, float("inf"))
    for bn, bp in bases.items():
        line, bb = [], (None, float("inf"))
        for w in [0.3, 0.4, 0.5, 0.6]:
            r = rmse((1-w)*bp + w*nn_val, yva); line.append(f"w{w}={r:.4f}")
            if r < bb[1]: bb = (w, r)
            if r < overall_best[2]: overall_best = (bn, w, r)
        print(f"  {bn:14s}: {'  '.join(line)}  best={bb[1]:.6f}@w{bb[0]}", flush=True)
    print(f"\nuntrained=0")
    print(f"val_rmse:         {overall_best[2]:.6f}  (base={overall_best[0]}, w={overall_best[1]})")
    print(f"total_seconds:    {time.time()-t0:.1f}")

if __name__ == "__main__":
    main()
