# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data loading, normalization, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `T:/clipforge/datasets/kinetics/labelled/` contains parquet+json pairs. Run `uv run prepare.py` to verify loading.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Task

**Deadlift form regression**: Train a model that predicts 11 form metric scores (0-9) from pose/DensePose features extracted from video frames.

**Data**: 23 labelled samples in `T:/clipforge/datasets/kinetics/labelled/`. Each sample = 75 frames × 334 features + sex → 11 integer scores.

**Key constraint**: 23 samples with 25K input features. Overfitting is guaranteed with complex models. The baseline uses temporal aggregation (mean/std/min/max) to reduce to ~1340 features, then a tiny MLP with heavy dropout and weight decay.

## Experimentation

Each experiment runs on a single GPU (or CPU — with 23 samples it's fast either way). The training script runs for a **fixed time budget of 2 minutes** (wall clock training time). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, feature engineering, regularization, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and training constants.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_rmse` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_rmse.** Lower is better. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the feature engineering, the regularization strategy. The only constraint is that the code runs without crashing and finishes within the time budget.

**Experiment directions to explore:**
- **Architecture**: Ridge regression, Lasso, ElasticNet, wider/narrower MLP, 1D CNN on raw frames
- **Temporal features**: velocity (frame diffs), acceleration, range of motion, peak angles
- **Augmentation**: Gaussian noise on features, temporal jitter, mixup, feature dropout
- **Regularization**: dropout, weight decay, L1 penalty, feature selection
- **Feature selection**: PCA, mutual information, LASSO-based selection
- **Per-metric models**: train 11 separate models instead of one multi-output model
- **Loss**: MSE vs Huber vs per-output weighting
- **sklearn baselines**: Ridge, Lasso, ElasticNet on aggregated features (may outperform neural nets at 23 samples)

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_rmse:         1.234567
training_seconds: 2.1
total_seconds:    2.5
num_epochs:       842
num_params:       90123
best_epoch:       642
```

You can extract the key metric from the log file:

```
grep "^val_rmse:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_rmse	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_rmse achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB (0.0 for CPU runs or crashes)
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_rmse	memory_gb	status	description
a1b2c3d	1.834567	0.0	keep	baseline
b2c3d4e	1.723456	0.0	keep	increase weight decay to 0.2
c3d4e5f	1.950000	0.0	discard	switch to MSE loss
d4e5f6g	0.000000	0.0	crash	bad feature engineering
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar9`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_rmse:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv
8. If val_rmse improved (lower), you "advance" the branch, keeping the git commit
9. If val_rmse is equal or worse, you git reset back to where you started

**Timeout**: Each experiment should take under 3 minutes total. If a run exceeds 5 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes, use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical approaches, try sklearn models. The loop runs until the human interrupts you, period.
