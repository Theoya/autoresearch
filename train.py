"""
Deadlift form regression training script.
Temporal aggregation + MLP baseline for 23 labelled samples.
Usage: uv run train.py
"""

import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    NUM_FRAMES, NUM_FEATURES_PER_FRAME, NUM_OUTPUTS, TIME_BUDGET,
    METRIC_NAMES, load_all_data, make_splits, normalize, evaluate_rmse,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

HIDDEN_DIM = 64
DROPOUT = 0.5
LR = 1e-3
WEIGHT_DECAY = 0.1
MAX_EPOCHS = 5000
PATIENCE = 200
EVAL_EVERY = 10
SEED = 42

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

NUM_INPUT_FEATURES = (NUM_FEATURES_PER_FRAME + 1) * 4  # mean/std/min/max of 335 features = 1340

class DeadliftFormMLP(nn.Module):
    """Temporal aggregation + 2-layer MLP for form regression."""

    def __init__(self, input_dim=NUM_INPUT_FEATURES, hidden_dim=HIDDEN_DIM,
                 output_dim=NUM_OUTPUTS, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 75, 335) normalized pose features
        Returns:
            (B, 11) predicted form scores
        """
        # Temporal aggregation: mean, std, min, max across frames
        x_mean = x.mean(dim=1)       # (B, 335)
        x_std = x.std(dim=1)         # (B, 335)
        x_min = x.min(dim=1).values  # (B, 335)
        x_max = x.max(dim=1).values  # (B, 335)
        agg = torch.cat([x_mean, x_std, x_min, x_max], dim=1)  # (B, 1340)
        return self.net(agg)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(SEED)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load data
X, y, ids = load_all_data()
print(f"Loaded {len(ids)} samples")

X_train, X_val, y_train, y_val, train_idx, val_idx = make_splits(X, y)
print(f"Train: {len(X_train)}, Val: {len(X_val)}")

X_train, X_val, mean, std = normalize(X_train, X_val)

# Move to device
X_train = X_train.to(device)
X_val = X_val.to(device)
y_train = y_train.to(device)
y_val = y_val.to(device)

# Model
model = DeadliftFormMLP().to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Loss: Huber (smooth L1) — more robust than MSE with few samples
criterion = nn.HuberLoss(delta=1.0)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
best_val_rmse = float("inf")
best_epoch = 0
best_state = None
epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):
    # Check time budget
    elapsed = time.time() - t_start_training
    if elapsed >= TIME_BUDGET:
        print(f"\nTime budget ({TIME_BUDGET}s) reached at epoch {epoch}")
        break

    # Train step (full batch)
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()

    # Evaluate
    if epoch % EVAL_EVERY == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_rmse = evaluate_rmse(val_pred, y_val)
            train_pred = model(X_train)
            train_rmse = evaluate_rmse(train_pred, y_train)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += EVAL_EVERY

        if epoch % (EVAL_EVERY * 10) == 0:
            print(f"  epoch {epoch:5d} | train_rmse: {train_rmse:.6f} | val_rmse: {val_rmse:.6f} | best: {best_val_rmse:.6f} @ {best_epoch}")

        # Early stopping
        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

# Restore best weights
if best_state is not None:
    model.load_state_dict(best_state)

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
t_end_training = time.time()
training_seconds = t_end_training - t_start_training

with torch.no_grad():
    val_pred = model(X_val)
    val_rmse = evaluate_rmse(val_pred, y_val)

    # Per-metric RMSE
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
print(f"num_epochs:       {epoch}")
print(f"num_params:       {num_params}")
print(f"best_epoch:       {best_epoch}")
