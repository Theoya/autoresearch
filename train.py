"""
Deadlift form regression training script.
Per-metric independent tiny MLPs on temporally aggregated features.
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
PATIENCE = 300
EVAL_EVERY = 10
SEED = 42

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

# ---------------------------------------------------------------------------
# Train one MLP per metric
# ---------------------------------------------------------------------------

t_start_training = time.time()
best_models = []
total_params = 0
total_epochs = 0

for metric_idx, metric_name in enumerate(METRIC_NAMES):
    torch.manual_seed(SEED + metric_idx)

    model = SingleMetricMLP().to(device)
    total_params += sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.HuberLoss(delta=1.0)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        elapsed = time.time() - t_start_training
        if elapsed >= TIME_BUDGET * 0.9:
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
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += EVAL_EVERY

            if epochs_without_improvement >= PATIENCE:
                break

        total_epochs += 1

    if best_state is not None:
        model.load_state_dict(best_state)
    best_models.append(model)
    print(f"  {metric_name:25s}: best_epoch={best_epoch:5d}  val_mse={best_val_loss:.4f}")

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

t_end_training = time.time()
training_seconds = t_end_training - t_start_training

val_preds = []
for model in best_models:
    model.eval()
    with torch.no_grad():
        val_preds.append(model(X_val_agg))

val_pred = torch.stack(val_preds, dim=1)  # (N_val, 11)
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
