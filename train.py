"""
Deadlift form regression training script.
Best-k ensemble of per-metric tiny MLPs — train many, keep the best.
Usage: uv run train.py
"""

import time
import copy

import torch
import torch.nn as nn

from prepare import (
    NUM_FRAMES, NUM_FEATURES_PER_FRAME, NUM_OUTPUTS, TIME_BUDGET,
    METRIC_NAMES, load_all_data, make_splits, normalize, evaluate_rmse,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HIDDEN_DIM = 32
DROPOUT = 0.5
LR = 1e-3
WEIGHT_DECAY = 0.1
MAX_EPOCHS = 5000
PATIENCE = 1000
EVAL_EVERY = 2
BASE_SEED = 42
NUM_SEEDS = 8   # train this many per metric
TOP_K = 1       # keep best k by val loss for ensemble

# ---------------------------------------------------------------------------
# Model: one small MLP per metric
# ---------------------------------------------------------------------------

NUM_INPUT_FEATURES = (NUM_FEATURES_PER_FRAME + 1) * 4  # 1340

class SingleMetricMLP(nn.Module):
    def __init__(self, input_dim=NUM_INPUT_FEATURES, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Xavier uniform init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ---------------------------------------------------------------------------
# Temporal aggregation
# ---------------------------------------------------------------------------

def aggregate(X):
    """(N, 75, 335) -> (N, 1340) via mean/std/min/max across frames."""
    x_mean = X.mean(dim=1)
    x_std = X.std(dim=1)
    x_min = X.min(dim=1).values
    x_max = X.max(dim=1).values
    return torch.cat([x_mean, x_std, x_min, x_max], dim=1)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

X, y, ids = load_all_data()
print(f"Loaded {len(ids)} samples")

X_train, X_val, y_train, y_val, train_idx, val_idx = make_splits(X, y)
print(f"Train: {len(X_train)}, Val: {len(X_val)}")

X_train, X_val, mean, std = normalize(X_train, X_val)

X_train_agg = aggregate(X_train).to(device)
X_val_agg = aggregate(X_val).to(device)
y_train = y_train.to(device)
y_val = y_val.to(device)

print(f"Aggregated features: {X_train_agg.shape[1]}")
print(f"Training {NUM_SEEDS} seeds per metric, keeping top {TOP_K}")

# ---------------------------------------------------------------------------
# Train many seeds per metric, select best-k
# ---------------------------------------------------------------------------

t_start_training = time.time()
all_models = []
total_params = 0
total_epochs = 0

for metric_idx, metric_name in enumerate(METRIC_NAMES):
    candidates = []  # (val_loss, model_state)

    for seed_offset in range(NUM_SEEDS):
        elapsed = time.time() - t_start_training
        if elapsed >= TIME_BUDGET * 0.85:
            break

        seed = BASE_SEED + metric_idx * 100 + seed_offset
        torch.manual_seed(seed)

        model = SingleMetricMLP().to(device)
        if metric_idx == 0 and seed_offset == 0:
            params_per_model = sum(p.numel() for p in model.parameters())

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
        criterion = nn.HuberLoss(delta=2.5)

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            elapsed = time.time() - t_start_training
            if elapsed >= TIME_BUDGET * 0.85:
                break

            model.train()
            optimizer.zero_grad()
            pred = model(X_train_agg)
            loss = criterion(pred, y_train[:, metric_idx])
            loss.backward()
            optimizer.step()

            if epoch % EVAL_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val_agg)
                    val_loss = ((val_pred - y_val[:, metric_idx]) ** 2).mean().item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += EVAL_EVERY

                if epochs_without_improvement >= PATIENCE:
                    break

            total_epochs += 1

        if best_state is not None:
            candidates.append((best_val_loss, best_state))

    # Select top-k by val loss
    candidates.sort(key=lambda x: x[0])
    top_k = candidates[:TOP_K]

    metric_models = []
    for val_loss, state in top_k:
        m = SingleMetricMLP().to(device)
        m.load_state_dict(state)
        metric_models.append(m)

    all_models.append(metric_models)
    total_params += params_per_model * len(metric_models)
    n_trained = len(candidates)
    best_loss = top_k[0][0] if top_k else float('inf')
    print(f"  {metric_name:25s}: trained {n_trained}/{NUM_SEEDS}, kept {len(top_k)}, best_val_mse={best_loss:.4f}")

# ---------------------------------------------------------------------------
# Final evaluation — average top-k predictions
# ---------------------------------------------------------------------------

t_end_training = time.time()
training_seconds = t_end_training - t_start_training

val_preds_per_metric = []
for metric_idx, metric_models in enumerate(all_models):
    if not metric_models:
        # Fallback: predict training mean
        val_preds_per_metric.append(y_train[:, metric_idx].mean().expand(y_val.shape[0]))
        continue
    seed_preds = []
    for model in metric_models:
        model.eval()
        with torch.no_grad():
            seed_preds.append(model(X_val_agg))
    val_preds_per_metric.append(torch.stack(seed_preds).mean(dim=0))

val_pred = torch.stack(val_preds_per_metric, dim=1)
val_rmse = evaluate_rmse(val_pred, y_val)

print("\nPer-metric RMSE:")
for i, name in enumerate(METRIC_NAMES):
    metric_rmse = ((val_pred[:, i] - y_val[:, i]) ** 2).mean().sqrt().item()
    print(f"  {name:25s}: {metric_rmse:.4f}")

t_end = time.time()
total_seconds = t_end - t_start

print("\n---")
print(f"val_rmse:         {val_rmse:.6f}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"total_seconds:    {total_seconds:.1f}")
print(f"num_epochs:       {total_epochs}")
print(f"num_params:       {total_params}")
print(f"best_epoch:       0")
