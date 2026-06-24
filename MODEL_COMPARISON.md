# Model Comparison: claude_dataset impact + autoresearch improvement loop (jun24)

Branch: `autoresearch/jun24`. Metric: **val_rmse** (lower is better), single train/val split
defined in the read-only `prepare.py` / `prepare_squat.py` (`evaluate_rmse`, val_ratio=0.2).
All training runs on a single RTX 3080 GPU within the fixed in-script time budget (102s of
training, `TIME_BUDGET=120` x 0.85). Feature extraction used the Form-Improver DensePose+MediaPipe
extractor (25,051-col parquet per clip: sex + 75 frames x 334 features).

## Summary table

| Model | Data | Samples | val_rmse | Notes |
|---|---|---:|---:|---|
| **Deadlift — existing model (BASELINE)** | old only | 78 | **1.878670** | per-metric MLPs, h64, 30 seeds, top-1, 9-stat. All 11 metrics trained. |
| **Deadlift — existing model + claude data** | old + new | 152 | **1.688505** | identical code, +74 claude_dataset clips. **More data: −0.190 (−10.1%).** |
| **Deadlift — best improved** | old + new | 152 | **1.609103** | D11: h64, top-3 ensemble, patience 40, 7-stat. **−0.079 vs old+new; −0.270 vs baseline.** All 11 metrics train (untrained=0). |
| **Squat — baseline (fresh, no prior model)** | new only | 57 | **0.624933** | per-metric MLPs, h64, 30 seeds, top-1, 9-stat, patience 150. All 10 metrics trained. (mean-pred ref = 0.5686) |
| **Squat — best improved** | new only | 57 | **0.5367** | S2: h128, top-3 ensemble, patience 40, 7-stat. **−0.088 vs squat baseline (−14%).** All 10 metrics train. |

## Did adding claude_dataset data help? (the core question)

**Yes — clearly, for deadlift.** Holding the model code fixed and only adding the 74 new
claude_dataset deadlift clips (78 → 152 samples) lowered val_rmse from **1.878670 → 1.688505**,
a **−0.190 (10.1%) improvement**. This is the cleanest controlled comparison: same architecture,
same hyperparameters, same evaluation — only the training set grew.

This **confirms the prediction in `unified_summary.md`**, which concluded that the limiting factor
was the tiny dataset (originally 23 samples) and that "the unified approach would likely become
competitive with more training data." We did not need the unified model to see the data benefit —
more data helped the existing per-metric approach directly, and the larger dataset also made the
models train *faster to convergence* (D1 h128 completed all 11 metrics in budget where it would not
have on 78 samples), opening room for higher-capacity / ensembled configs.

## Improvement loop findings

Deadlift (on 152 samples): the wins that generalized and **train all 11 metrics honestly** were
(1) **top-3 ensemble** averaging and (2) **patience 40** + **7-stat aggregation** (simpler features).
hidden_dim 256, dropout 0.3, weight_decay 0.01, and top-5 ensemble all regressed and were reverted.

A key methodology lesson surfaced mid-loop: configs that crank `NUM_SEEDS` high (50/70) appear to
"improve" val_rmse but actually **starve the later metrics** — the 102s training cap is reached
before metrics like `controlledDescent`/`lockoutPosition` get any seeds, so they silently fall back
to mean-prediction. Mean-prediction on those high-variance metrics happens to score *low* RMSE, so
the headline number drops while the model is strictly worse. These degenerate runs (D8, D9, D12)
were detected via an `untrained=` check (count of `trained 0/N` per-metric lines) and **reverted**.
The accepted best (D11, 1.609103) trains **all 11 metrics** (`untrained=0`).

Squat: the same D11-style recipe (top-3 + patience 40 + 7-stat, then h128) transferred well,
improving the fresh baseline by ~14% to **0.5367** with all 10 metrics trained. Squat scores are
tightly clustered (mean-pred ref RMSE 0.5686), so absolute RMSE is much lower than deadlift.

## Files
- `train.py` — deadlift trainer (D11 config, + feature cache). `prepare.py` unchanged (read-only).
- `train_squat.py` / `prepare_squat.py` — squat trainer + data prep (DATA_DIR=`claude_squat_labelled`, 10 metrics).
- `results.tsv` — full experiment log (baseline, old+new, D1–D12, squat baseline, S1–S4).
- New deadlift features added to `kinetics/labelled/` as `claude_<id>.{parquet,json}` (74 pairs;
  list in `datasets/_claude_stage/added_to_labelled.txt`). Originals untouched.

## Feature extraction
- Deadlift: **74/74** claude clips extracted successfully (0 failures).
- Squat: **57/57** claude clips extracted successfully (0 failures).
- Validated one clip first: 1-row x 25,051-col parquet, 334 features/frame, no NaN/Inf.
