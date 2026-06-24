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
| Deadlift — best improved (cycle 1) | old + new | 152 | 1.609103 | D11: h64, top-3 ensemble, patience 40, 7-stat (neural only). |
| **Deadlift — BEST (cycle 2)** | old + new | 152 | **1.499465** | **0.7·RidgeCV + 0.3·neural-ensemble blend** (7-stat). **−0.110 vs D11; −0.379 vs baseline.** untrained=0. |
| **Squat — baseline (fresh, no prior model)** | new only | 57 | **0.624933** | per-metric MLPs, h64, 30 seeds, top-1, 9-stat, patience 150. (mean-pred ref = 0.5686) |
| Squat — best improved (cycle 1) | new only | 57 | 0.5367 | S2: h128, top-3 ensemble, patience 40, 7-stat (neural only). |
| **Squat — BEST (cycle 2)** | new only | 57 | **0.489028** | **0.5·RidgeCV + 0.5·neural-ensemble blend** (h128, 7-stat). **−0.048 vs S2 (−9%); −0.136 vs baseline.** untrained=0. |

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

## Cycle 2 findings — what worked

The big lever was **per-metric `RidgeCV` linear models** on the 7-stat features. They fit in
milliseconds, so all metrics are modelled honestly with zero risk of the seed-starvation
degeneracy. Per-metric Ridge alone scored **1.5116 (deadlift)** / **0.5300 (squat)** — already
beating the cycle-1 neural bests (1.6091 / 0.5367).

Then **blending Ridge with the neural ensemble** (diverse model families) won outright:
- Deadlift: `0.7·Ridge + 0.3·neural` → **1.499465** (the neural net is the weaker but decorrelated member).
- Squat: `0.5·Ridge + 0.5·neural` → **0.489028** (equal blend; the two are similarly strong here).
Both blend curves are smooth with a flat minimum (deadlift 0.2–0.4, squat 0.4–0.6), so the chosen
weights are robust rather than over-fit to the 30/8-sample val set.

What did **not** help (reverted): Lasso, ElasticNet (slightly worse than Ridge); SelectKBest
feature selection (monotonically worse as k shrinks — L2 handles the wide feature space better than
hard selection); PCA (≈ tied with full Ridge, kept full Ridge for simplicity); finer/LOO alpha
grids (identical result — the optimal alpha is interior to the grid).

**Gap to the historical 0.599:** that number was on the original 23-sample task with a different
(easier) split. On the current 152-sample split the comparable "existing model" baseline is 1.879;
cycle 2 closes ~46% of the baseline→0 distance to **1.499**. The remaining gap is largely the
harder, larger val set (30 samples spanning more form variation) plus genuinely hard high-variance
metrics (controlledDescent, lockoutPosition).

`train.py` / `train_squat.py` now emit `nn_rmse` / `ridge_rmse` / `blend_rmse` so the blend is
auditable each run, and both keep the `untrained=` honesty property (Ridge always fits all metrics).

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
