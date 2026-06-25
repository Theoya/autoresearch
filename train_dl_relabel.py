"""
DEADLIFT form regression on RE-LABELED clips only (consistent grounded-rubric labels).
Per-metric tiny-MLP ensemble blended with per-metric ExtraTrees. Mirror of train_bench.py
importing prepare_dl_relabel (11 metrics). Usage: uv run train_dl_relabel.py
"""
import os, glob, time, copy, hashlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesRegressor

from prepare_dl_relabel import (
    NUM_FRAMES, NUM_FEATURES_PER_FRAME, NUM_OUTPUTS, TIME_BUDGET,
    METRIC_NAMES, DATA_DIR, load_all_data, make_splits, normalize, evaluate_rmse,
)


def load_all_data_cached():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "claude_*.parquet")))
    key = hashlib.md5(("|".join(os.path.basename(f) for f in files)).encode()).hexdigest()[:12]
    cache_dir = os.path.join(".cache"); os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"dlrl_{len(files)}_{key}.pt")
    if os.path.exists(cache_path):
        blob = torch.load(cache_path)
        return blob["X"], blob["y"], blob["ids"]
    X, y, ids = load_all_data()
    torch.save({"X": X, "y": y, "ids": ids}, cache_path)
    return X, y, ids

HIDDEN_DIM, DROPOUT, LR, WEIGHT_DECAY = 128, 0.5, 1e-3, 0.02
MAX_EPOCHS, PATIENCE, EVAL_EVERY, BASE_SEED, NUM_SEEDS, TOP_K = 5000, 40, 2, 42, 30, 3
BLEND_W = 0.5
ET_KW = dict(n_estimators=300, max_features=0.2, min_samples_leaf=3, random_state=42, n_jobs=-1)
NUM_INPUT_FEATURES = (NUM_FEATURES_PER_FRAME + 1) * 7


class SingleMetricMLP(nn.Module):
    def __init__(self, input_dim=NUM_INPUT_FEATURES, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.nonlinear = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                       nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.shortcut = nn.Linear(input_dim, 1)
        for m in [*self.nonlinear, self.shortcut]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return (self.nonlinear(x) + self.shortcut(x)).squeeze(-1)


def aggregate(X):
    return torch.cat([X.mean(dim=1), X.std(dim=1), X.min(dim=1).values, X.max(dim=1).values,
                      X.median(dim=1).values, X.quantile(0.25, dim=1), X.quantile(0.75, dim=1)], dim=1)


t_start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
X, y, ids = load_all_data_cached()
print(f"Loaded {len(ids)} samples")
X_train, X_val, y_train, y_val, _, _ = make_splits(X, y)
print(f"Train: {len(X_train)}, Val: {len(X_val)}")
X_train, X_val, mean, std = normalize(X_train, X_val)
X_train_agg = aggregate(X_train).to(device); X_val_agg = aggregate(X_val).to(device)
y_train = y_train.to(device); y_val = y_val.to(device)

t0 = time.time(); all_models = []; total_params = 0; total_epochs = 0
for metric_idx, metric_name in enumerate(METRIC_NAMES):
    candidates = []
    for seed_offset in range(NUM_SEEDS):
        if time.time() - t0 >= TIME_BUDGET * 0.85: break
        torch.manual_seed(BASE_SEED + metric_idx * 100 + seed_offset)
        model = SingleMetricMLP().to(device)
        if metric_idx == 0 and seed_offset == 0:
            params_per_model = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
        crit = nn.MSELoss(); best, best_state, noimp = float("inf"), None, 0
        for epoch in range(1, MAX_EPOCHS + 1):
            if time.time() - t0 >= TIME_BUDGET * 0.85: break
            model.train(); opt.zero_grad()
            loss = crit(model(X_train_agg), y_train[:, metric_idx]); loss.backward(); opt.step()
            if epoch % EVAL_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    vl = ((model(X_val_agg) - y_val[:, metric_idx]) ** 2).mean().item()
                if vl < best: best, best_state, noimp = vl, copy.deepcopy(model.state_dict()), 0
                else: noimp += EVAL_EVERY
                if noimp >= PATIENCE: break
            total_epochs += 1
        if best_state is not None: candidates.append((best, best_state))
    candidates.sort(key=lambda x: x[0]); top_k = candidates[:TOP_K]
    mm = []
    for _, state in top_k:
        m = SingleMetricMLP().to(device); m.load_state_dict(state); mm.append(m)
    all_models.append(mm); total_params += params_per_model * len(mm)
    print(f"  {metric_name:20s}: kept {len(top_k)}, best_val_mse={top_k[0][0]:.4f}" if top_k else f"  {metric_name}: none")

train_secs = time.time() - t0
vpm = []
for mi, mm in enumerate(all_models):
    if not mm: vpm.append(y_train[:, mi].mean().expand(y_val.shape[0])); continue
    preds = []
    for model in mm:
        model.eval()
        with torch.no_grad(): preds.append(model(X_val_agg))
    vpm.append(torch.stack(preds).mean(dim=0))
nn_val = torch.stack(vpm, dim=1); nn_rmse = evaluate_rmse(nn_val, y_val)

Xtr_np, Xva_np, ytr_np = X_train_agg.cpu().numpy(), X_val_agg.cpu().numpy(), y_train.cpu().numpy()
tree_val = np.zeros((y_val.shape[0], NUM_OUTPUTS), dtype=np.float32)
for i in range(NUM_OUTPUTS):
    tree_val[:, i] = ExtraTreesRegressor(**ET_KW).fit(Xtr_np, ytr_np[:, i]).predict(Xva_np)
tree_val_t = torch.tensor(tree_val, device=y_val.device); tree_rmse = evaluate_rmse(tree_val_t, y_val)
val_pred = (1.0 - BLEND_W) * tree_val_t + BLEND_W * nn_val; val_rmse = evaluate_rmse(val_pred, y_val)

print(f"\nnn_rmse:    {nn_rmse:.6f}")
print(f"tree_rmse:  {tree_rmse:.6f}  (ExtraTrees base)")
print(f"blend_rmse: {val_rmse:.6f}  (W={BLEND_W})")
print("\nPer-metric RMSE (blend):")
for i, name in enumerate(METRIC_NAMES):
    print(f"  {name:20s}: {((val_pred[:, i] - y_val[:, i]) ** 2).mean().sqrt().item():.4f}")
print(f"\nval_rmse:         {val_rmse:.6f}")
print(f"training_seconds: {train_secs:.1f}")
print(f"num_params:       {total_params}")
