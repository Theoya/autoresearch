"""
3-class SEX/GENDER CLASSIFIER (male / female / unclear) training + honest CV evaluation.

Same recipe as train_classifier.py (the exercise classifier). Label = the `gender` Claude
Code assigned during dataset labeling (manifest field). Features = MediaPipe+limb+angle pose
columns aggregated over 75 frames; DensePose channels are skipped (ablation noise).

Eval = stratified K-fold cross-validation (unclear is the minority class at ~33, so stratify),
with class weighting for the heavy male skew. Reports overall accuracy, macro-F1, per-class
precision/recall/F1, a 3x3 confusion matrix, AND the majority-class (predict-all-male) baseline
for honest context. No leakage: the StandardScaler is fit INSIDE each fold on train only.

Candidate classifiers: LogisticRegression (class_weight='balanced'), RandomForest, ExtraTrees,
small MLP. Picks the best honest one by macro-F1 (ties broken by accuracy).

Usage: uv run train_sex.py
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

from prepare_sex import (
    CLASS_NAMES, load_all_data_cached, aggregate, group_frame_mask, frame_feature_names,
)

SEED = 42
N_FOLDS = 5
DEFAULT_FEATURE_GROUP = "mp"  # MediaPipe+limb+angle, no DensePose


def make_clf(name):
    if name.startswith("logreg"):
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


def majority_baseline(y):
    """Predict-all-majority-class baseline (honest context for the heavy male skew)."""
    maj = int(np.bincount(y).argmax())
    oof = np.full(len(y), maj, dtype=int)
    acc = accuracy_score(y, oof)
    mf1 = f1_score(y, oof, average="macro")
    prec, rec, f1, sup = precision_recall_fscore_support(y, oof, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y, oof, labels=[0, 1, 2])
    return dict(name=f"baseline(all={CLASS_NAMES[maj]})", acc=acc, macro_f1=mf1,
                prec=prec, rec=rec, f1=f1, sup=sup, cm=cm, maj=maj)


def print_report(r):
    print(f"\n=== {r['name']} ===")
    print(f"accuracy:  {r['acc']:.4f}    macro-F1: {r['macro_f1']:.4f}")
    print(f"{'class':10s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'n':>4s}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:10s} {r['prec'][i]:6.3f} {r['rec'][i]:6.3f} {r['f1'][i]:6.3f} {r['sup'][i]:4d}")
    print("confusion matrix (rows=true, cols=pred)  order: male, female, unclear")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:9s} " + " ".join(f"{v:3d}" for v in r["cm"][i]))


def top_features(name, X, y, feat_names, k=15):
    """For an interpretable winner, surface the highest-|coef| / importance features.

    Helps answer: do body-proportion (limb_/angle_) features carry the weight?
    """
    clf = make_clf(name)
    clf.fit(X, y)
    est = clf[-1] if hasattr(clf, "__getitem__") else clf
    if hasattr(est, "coef_"):
        imp = np.abs(est.coef_).mean(axis=0)  # mean |coef| across the 3 one-vs-rest rows
        kind = "mean|coef|"
    elif hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
        kind = "importance"
    else:
        return None
    order = np.argsort(imp)[::-1][:k]
    return kind, [(feat_names[i], float(imp[i])) for i in order]


def main():
    t0 = time.time()
    X3, y_t, ids = load_all_data_cached()
    y = y_t.numpy()
    print(f"Loaded {len(y)} clips")
    print("class counts:", {CLASS_NAMES[i]: int((y == i).sum()) for i in range(3)})
    from collections import Counter
    print("by exercise:", dict(Counter(ex for ex, _ in ids)))
    print(f"{N_FOLDS}-fold stratified CV  (scaler fit inside each fold, class_weight=balanced)")

    # default feature set = MediaPipe+limb+angle (no DensePose)
    mask = group_frame_mask(DEFAULT_FEATURE_GROUP)
    X = aggregate(X3[:, :, mask]).numpy().astype(np.float64)
    print(f"feature group '{DEFAULT_FEATURE_GROUP}': {int(mask.sum())} per-frame channels "
          f"-> aggregated dim {X.shape[1]}")

    # ----- honest majority baseline -----
    base = majority_baseline(y)
    print_report(base)

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

    best = max(results, key=lambda r: (round(r["macro_f1"], 6), round(r["acc"], 6)))
    print("\n" + "=" * 60)
    print(f"BEST: {best['name']}  accuracy={best['acc']:.4f}  macro-F1={best['macro_f1']:.4f}")
    print(f"vs majority baseline: acc={base['acc']:.4f}  macro-F1={base['macro_f1']:.4f}")
    print("=" * 60)

    # ----- Feature-group ablation: MediaPipe-only vs full vs DensePose-only -----
    print(f"\n--- Feature-group ablation (winning model = {best['name']}) ---")
    ablation = {}
    for grp, label in [("all", "full(334)"), ("mp", "mediapipe(102)"), ("dp", "densepose(232)")]:
        m = group_frame_mask(grp)
        Xg = aggregate(X3[:, :, m]).numpy().astype(np.float64)
        rg = evaluate(best["name"], Xg, y)
        ablation[grp] = {"label": label, "feat_dim": int(Xg.shape[1]),
                         "accuracy": round(float(rg["acc"]), 6),
                         "macro_f1": round(float(rg["macro_f1"]), 6),
                         "per_class_recall": {CLASS_NAMES[i]: round(float(rg["rec"][i]), 3) for i in range(3)},
                         "confusion_matrix": rg["cm"].tolist()}
        print(f"  {label:16s} dim={Xg.shape[1]:5d}  acc={rg['acc']:.4f}  macroF1={rg['macro_f1']:.4f}  "
              f"female_recall={rg['rec'][1]:.3f}")

    # ----- which features carry the weight? (body proportion vs landmark coords) -----
    feat_names_per_frame = [n for n in frame_feature_names() if n.startswith(("mp_", "limb_", "angle_"))]
    agg_stats = ["mean", "std", "min", "max", "median", "q25", "q75"]
    agg_feat_names = [f"{s}:{n}" for s in agg_stats for n in feat_names_per_frame]
    tf = top_features(best["name"], X, y, agg_feat_names, k=20)
    top_feats_out = None
    if tf:
        kind, feats = tf
        print(f"\n--- Top {len(feats)} features for {best['name']} ({kind}) ---")
        for fn, v in feats:
            print(f"  {v:8.4f}  {fn}")
        # tally which prefix groups dominate the top-50
        tf_all = top_features(best["name"], X, y, agg_feat_names, k=50)
        grp_count = Counter()
        for fn, _ in tf_all[1]:
            base_name = fn.split(":", 1)[1]
            grp_count[base_name.split("_")[0]] += 1
        print(f"  top-50 feature group tally: {dict(grp_count)}")
        top_feats_out = {"kind": kind, "top20": [[fn, round(v, 5)] for fn, v in feats],
                         "top50_group_tally": dict(grp_count)}

    # save summary
    save_dir = os.path.join("models", "sex")
    os.makedirs(save_dir, exist_ok=True)
    summary = {
        "task": "sex_classifier",
        "classes": CLASS_NAMES,
        "n_samples": int(len(y)),
        "class_counts": {CLASS_NAMES[i]: int((y == i).sum()) for i in range(3)},
        "by_exercise": dict(Counter(ex for ex, _ in ids)),
        "feature_group": DEFAULT_FEATURE_GROUP,
        "feature_dim": int(X.shape[1]),
        "n_folds": N_FOLDS,
        "best_model": best["name"],
        "majority_baseline": {
            "predict_all": CLASS_NAMES[base["maj"]],
            "accuracy": round(float(base["acc"]), 6),
            "macro_f1": round(float(base["macro_f1"]), 6),
        },
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
        "feature_group_ablation": ablation,
        "top_features": top_feats_out,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {save_dir}/config.json")

    # ----- Persist artifact: winning model refit on ALL data, MediaPipe feature group -----
    import joblib
    final = make_clf(best["name"])
    final.fit(X, y)
    artifact = {
        "pipeline": final,
        "classes": CLASS_NAMES,
        "model_name": best["name"],
        "feature_group": DEFAULT_FEATURE_GROUP,
        "agg_stats": agg_stats,
        "cv_accuracy": round(float(best["acc"]), 6),
        "cv_macro_f1": round(float(best["macro_f1"]), 6),
        "baseline_accuracy": round(float(base["acc"]), 6),
        "baseline_macro_f1": round(float(base["macro_f1"]), 6),
    }
    joblib.dump(artifact, os.path.join(save_dir, "sex_clf_mp.joblib"))
    # also drop a copy next to the predictor in clipforge/pipeline/detect for the skill
    detect_dir = "T:/clipforge/pipeline/detect"
    if os.path.isdir(detect_dir):
        joblib.dump(artifact, os.path.join(detect_dir, "sex_clf_mp.joblib"))
        with open(os.path.join(detect_dir, "sex_metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Copied artifact + metrics to {detect_dir}/")
    print(f"Saved artifact to {save_dir}/sex_clf_mp.joblib  ({best['name']}, refit on all {len(y)} clips)")

    print("\n---")
    print(f"accuracy:         {best['acc']:.6f}")
    print(f"macro_f1:         {best['macro_f1']:.6f}")
    print(f"baseline_acc:     {base['acc']:.6f}")
    print(f"baseline_macrof1: {base['macro_f1']:.6f}")
    print(f"best_model:       {best['name']}")
    print(f"total_seconds:    {time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
