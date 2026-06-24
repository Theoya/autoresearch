"""
3-class EXERCISE CLASSIFIER (deadlift / squat / bench) training + honest CV evaluation.

Eval = stratified K-fold cross-validation (bench is the minority class at 22, so stratify).
Reports overall accuracy, macro-F1, per-class precision/recall/F1, and a 3x3 confusion matrix.
No leakage: the StandardScaler (and any selection) is fit INSIDE each fold on train only.

Candidate classifiers: LogisticRegression, RandomForest, ExtraTrees, small MLP.
Picks the best honest one by macro-F1 (ties broken by accuracy).

Usage: uv run train_classifier.py
"""
import os
import json
import time

import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
)

from prepare_classifier import CLASS_NAMES, load_all_data_cached, aggregate

SEED = 42
N_FOLDS = 5


def make_clf(name):
    if name.startswith("logreg"):
        # logreg, logreg_c0.1, logreg_c0.3, logreg_c3 ...
        C = float(name.split("_c")[1]) if "_c" in name else 1.0
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=5000, class_weight="balanced", random_state=SEED),
        )
    if name == "randomforest":
        return RandomForestClassifier(n_estimators=400, max_features="sqrt", min_samples_leaf=1,
                                      class_weight="balanced", random_state=SEED, n_jobs=-1)
    if name == "extratrees":
        return ExtraTreesClassifier(n_estimators=400, max_features="sqrt", min_samples_leaf=1,
                                    class_weight="balanced", random_state=SEED, n_jobs=-1)
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=1500,
                          early_stopping=False, random_state=SEED),
        )
    raise ValueError(name)


def evaluate(name, X, y):
    """Stratified K-fold OOF predictions; scaler fit inside each fold (pipeline handles it)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), -1, dtype=int)
    for tr, te in skf.split(X, y):
        clf = make_clf(name)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])
    assert (oof >= 0).all()
    acc = accuracy_score(y, oof)
    mf1 = f1_score(y, oof, average="macro")
    prec, rec, f1, sup = precision_recall_fscore_support(y, oof, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y, oof, labels=[0, 1, 2])
    return dict(name=name, acc=acc, macro_f1=mf1, prec=prec, rec=rec, f1=f1, sup=sup, cm=cm, oof=oof)


def print_report(r):
    print(f"\n=== {r['name']} ===")
    print(f"accuracy:  {r['acc']:.4f}    macro-F1: {r['macro_f1']:.4f}")
    print(f"{'class':10s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'n':>4s}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:10s} {r['prec'][i]:6.3f} {r['rec'][i]:6.3f} {r['f1'][i]:6.3f} {r['sup'][i]:4d}")
    print("confusion matrix (rows=true, cols=pred)  order: deadlift, squat, bench")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:9s} " + " ".join(f"{v:3d}" for v in r["cm"][i]))


def main():
    t0 = time.time()
    X3, y_t, ids = load_all_data_cached()
    y = y_t.numpy()
    X = aggregate(X3).numpy().astype(np.float64)
    print(f"Loaded {len(y)} clips, feature dim {X.shape[1]}")
    print("class counts:", {CLASS_NAMES[i]: int((y == i).sum()) for i in range(3)})
    print(f"{N_FOLDS}-fold stratified CV  (scaler fit inside each fold)")

    results = []
    candidates = ["logreg_c0.03", "logreg_c0.1", "logreg_c0.3", "logreg", "logreg_c3",
                  "randomforest", "extratrees", "mlp"]
    for name in candidates:
        t = time.time()
        r = evaluate(name, X, y)
        r["seconds"] = time.time() - t
        results.append(r)
        print_report(r)
        print(f"  ({r['seconds']:.1f}s)")

    # pick best by macro-F1, tie-break accuracy
    best = max(results, key=lambda r: (round(r["macro_f1"], 6), round(r["acc"], 6)))
    print("\n" + "=" * 60)
    print(f"BEST: {best['name']}  accuracy={best['acc']:.4f}  macro-F1={best['macro_f1']:.4f}")
    print("=" * 60)

    # save best model config + summary
    save_dir = os.path.join("models", "classifier")
    os.makedirs(save_dir, exist_ok=True)
    summary = {
        "task": "exercise_classifier",
        "classes": CLASS_NAMES,
        "n_samples": int(len(y)),
        "class_counts": {CLASS_NAMES[i]: int((y == i).sum()) for i in range(3)},
        "feature_dim": int(X.shape[1]),
        "n_folds": N_FOLDS,
        "best_model": best["name"],
        "results": {
            r["name"]: {
                "accuracy": round(float(r["acc"]), 6),
                "macro_f1": round(float(r["macro_f1"]), 6),
                "per_class": {CLASS_NAMES[i]: {"precision": round(float(r["prec"][i]), 4),
                                               "recall": round(float(r["rec"][i]), 4),
                                               "f1": round(float(r["f1"][i]), 4),
                                               "support": int(r["sup"][i])} for i in range(3)},
                "confusion_matrix": r["cm"].tolist(),
            } for r in results
        },
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {save_dir}/config.json")

    print("\n---")
    print(f"accuracy:         {best['acc']:.6f}")
    print(f"macro_f1:         {best['macro_f1']:.6f}")
    print(f"best_model:       {best['name']}")
    print(f"total_seconds:    {time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
