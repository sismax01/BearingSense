# =============================================================================
# BearingSense — Leak-Free Stratified Group CV & Diagnostic Metrics
# Engineered by : SISMAX (sismax01)
# =============================================================================
"""Leak-free evaluation, Stratified Group Cross-Validation, and diagnostic metrics.

Addresses a well-known methodological pitfall in bearing fault diagnosis research:
  - **Window-Level Data Leakage (Naive Random Split)**: Slicing a continuous vibration
    recording into hundreds of overlapping windows and randomly splitting those windows
    into train and test sets places adjacent windows from the *exact same physical test run*
    in both train and test. Because background noise, speed, load, and sensor mounting are
    identical within a run, models memorize run-specific artifacts and report artificially
    inflated accuracy.
  - **Run-Grouped Cross-Validation (Leak-Free Split)**: Groups all windows by their physical
    acquisition identifier (`run_id`) so that every window from a given recording is assigned
    exclusively to either the training set or the held-out test fold.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from bearingsense.features import FEATURE_NAMES
from bearingsense.modeling import (
    BaseBearingClassifier,
    GaussianDiagDiscriminant,
    PhysicsRuleThresholdClassifier,
    RandomForestBearingClassifier,
    RegularizedMultinomialLogistic,
)
from bearingsense.simulation import FAULT_CLASSES


def create_stratified_group_folds(
    df: pd.DataFrame,
    n_splits: int = 4,
    group_col: str = "run_id",
    label_col: str = "label",
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition dataset indices into leak-free Stratified Group K-Folds.

    Ensures:
      1. Zero group overlap (`set(train_groups) & set(test_groups) == set()`).
      2. Each fold contains held-out runs from every fault class whenever each class
         has at least `n_splits` distinct physical runs.
    """
    if group_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{group_col}' and '{label_col}' columns.")

    rng = np.random.default_rng(seed)

    run_df = df[[group_col, label_col]].drop_duplicates()
    fold_groups: List[List[str]] = [[] for _ in range(n_splits)]

    for cls_name in sorted(run_df[label_col].unique()):
        cls_runs = run_df[run_df[label_col] == cls_name][group_col].tolist()
        if len(cls_runs) < 2:
            raise ValueError(
                f"Class '{cls_name}' has only {len(cls_runs)} physical run(s). "
                "At least 2 independent runs per class are required for leak-free group splitting."
            )
        shuffled_runs = list(cls_runs)
        rng.shuffle(shuffled_runs)
        for idx, r_id in enumerate(shuffled_runs):
            fold_groups[idx % n_splits].append(r_id)

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    all_groups = df[group_col].to_numpy()
    all_indices = np.arange(len(df))

    for f_idx in range(n_splits):
        test_run_set = set(fold_groups[f_idx])
        if not test_run_set:
            continue
        test_mask = np.isin(all_groups, list(test_run_set))
        train_idx = all_indices[~test_mask]
        test_idx = all_indices[test_mask]

        # SISMAX Guardrail: Strict runtime verification of zero group leakage
        train_runs = set(all_groups[train_idx].tolist())
        test_runs = set(all_groups[test_idx].tolist())
        if train_runs & test_runs:
            raise RuntimeError("Data leakage detected: overlapping run_ids between train and test.")

        folds.append((train_idx, test_idx))

    return folds


def create_naive_random_window_folds(
    df: pd.DataFrame,
    n_splits: int = 4,
    label_col: str = "label",
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create naive stratified random window splits (intentionally ignoring `run_id`).

    Used strictly for diagnostic comparison to demonstrate how window-level leakage
    artificially inflates evaluation metrics when adjacent windows from the same run
    appear in both train and test sets.
    """
    rng = np.random.default_rng(seed)
    labels = df[label_col].to_numpy()
    fold_buckets: List[List[int]] = [[] for _ in range(n_splits)]

    for cls_name in sorted(df[label_col].unique()):
        cls_indices = np.where(labels == cls_name)[0].tolist()
        rng.shuffle(cls_indices)
        for i, idx in enumerate(cls_indices):
            fold_buckets[i % n_splits].append(idx)

    all_indices = np.arange(len(df))
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for f_idx in range(n_splits):
        test_idx = np.array(sorted(fold_buckets[f_idx]), dtype=int)
        train_mask = np.ones(len(df), dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_indices[train_mask]
        folds.append((train_idx, test_idx))

    return folds


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    y_prob: Optional[np.ndarray] = None,
    classes: Optional[Sequence[str]] = None,
) -> Dict:
    """Compute comprehensive multi-class evaluation metrics."""
    y_t = np.asarray(y_true, dtype=str)
    y_p = np.asarray(y_pred, dtype=str)
    if y_t.size == 0 or y_t.size != y_p.size:
        raise ValueError("y_true and y_pred must be non-empty and have equal length.")

    if classes is None:
        present = set(y_t.tolist()) | set(y_p.tolist())
        cls_list = sorted(present, key=lambda c: FAULT_CLASSES.index(c) if c in FAULT_CLASSES else 99)
    else:
        cls_list = list(classes)

    n_classes = len(cls_list)
    cls_to_idx = {c: i for i, c in enumerate(cls_list)}
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for gt, pr in zip(y_t, y_p):
        if gt in cls_to_idx and pr in cls_to_idx:
            cm[cls_to_idx[gt], cls_to_idx[pr]] += 1

    total = int(np.sum(cm))
    accuracy = float(np.trace(cm) / max(total, 1))

    per_class: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    supports: List[int] = []

    for i, c_name in enumerate(cls_list):
        tp = int(cm[i, i])
        fp = int(np.sum(cm[:, i]) - tp)
        fn = int(np.sum(cm[i, :]) - tp)
        tn = int(total - tp - fp - fn)
        support = tp + fn

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[c_name] = {
            "precision": prec,
            "recall": rec,
            "specificity": spec,
            "f1_score": f1,
            "support": float(support),
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.average(f1s, weights=supports)) if sum(supports) > 0 else 0.0

    brier_score = None
    if y_prob is not None and y_prob.shape == (len(y_t), n_classes):
        Y_onehot = np.zeros_like(y_prob, dtype=np.float64)
        for idx_row, gt in enumerate(y_t):
            if gt in cls_to_idx:
                Y_onehot[idx_row, cls_to_idx[gt]] = 1.0
        brier_score = float(np.mean(np.sum((y_prob - Y_onehot) ** 2, axis=1)))

    return {
        "accuracy": accuracy,
        "balanced_accuracy": macro_recall,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "brier_score": brier_score,
        "classes": cls_list,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "total_samples": total,
    }


def evaluate_models_cross_validation(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    n_splits: int = 4,
    seed: int = 42,
    compare_leaky_split: bool = True,
) -> Dict:
    """Evaluate all candidate models using leak-free Stratified Group K-Fold CV."""
    feat_names = feature_cols if feature_cols is not None else list(FEATURE_NAMES)
    X_all = df[feat_names].to_numpy(dtype=np.float64)
    y_all = df["label"].to_numpy(dtype=str)

    group_folds = create_stratified_group_folds(df, n_splits=n_splits, seed=seed)

    model_builders: List[Tuple[str, Callable[[], BaseBearingClassifier]]] = [
        ("Physics Rule Baseline", lambda: PhysicsRuleThresholdClassifier()),
        ("Logistic Regression (L2)", lambda: RegularizedMultinomialLogistic(l2_penalty=1.0)),
        ("Gaussian Discriminant", lambda: GaussianDiagDiscriminant(reg_var=0.15)),
        ("Random Forest", lambda: RandomForestBearingClassifier(n_estimators=60, max_depth=6, seed=seed)),
    ]

    results: Dict[str, Dict] = {}

    for model_name, builder in model_builders:
        oof_preds: List[str] = [""] * len(df)
        oof_probs = np.zeros((len(df), len(FAULT_CLASSES)), dtype=np.float64)
        fold_f1s: List[float] = []

        for train_idx, test_idx in group_folds:
            clf = builder()
            clf.fit(X_all[train_idx], y_all[train_idx], feature_names=feat_names)
            preds = clf.predict(X_all[test_idx])
            probs = clf.predict_proba(X_all[test_idx])

            for local_i, global_i in enumerate(test_idx):
                oof_preds[int(global_i)] = preds[local_i]
                for c_i, c_name in enumerate(clf.classes_):
                    if c_name in FAULT_CLASSES:
                        oof_probs[int(global_i), FAULT_CLASSES.index(c_name)] = probs[local_i, c_i]

            fold_metric = compute_classification_metrics(
                y_all[test_idx], preds, classes=FAULT_CLASSES
            )
            fold_f1s.append(fold_metric["macro_f1"])

        overall_metrics = compute_classification_metrics(
            y_all, oof_preds, y_prob=oof_probs, classes=FAULT_CLASSES
        )
        overall_metrics["fold_macro_f1_mean"] = float(np.mean(fold_f1s))
        overall_metrics["fold_macro_f1_std"] = float(np.std(fold_f1s))
        results[model_name] = overall_metrics

    leakage_experiment: Dict[str, Dict[str, float]] = {}
    if compare_leaky_split:
        td_cols = [c for c in feat_names if c.startswith("td_") or c.startswith("fd_")]
        X_td = df[td_cols].to_numpy(dtype=np.float64)
        naive_folds = create_naive_random_window_folds(df, n_splits=n_splits, seed=seed)

        for label_setting, X_mat, cols_used, folds_used in [
            ("Time/Freq Features (Leak-Free Run-Grouped Split)", X_td, td_cols, group_folds),
            ("Time/Freq Features (Naive Random Window Split — LEAKY)", X_td, td_cols, naive_folds),
            ("Full Envelope+Time Features (Leak-Free Run-Grouped Split)", X_all, feat_names, group_folds),
            ("Full Envelope+Time Features (Naive Random Window Split — LEAKY)", X_all, feat_names, naive_folds),
        ]:
            preds_setting = [""] * len(df)
            for tr_i, te_i in folds_used:
                rf = RandomForestBearingClassifier(n_estimators=50, max_depth=6, seed=seed)
                rf.fit(X_mat[tr_i], y_all[tr_i], feature_names=cols_used)
                p_fold = rf.predict(X_mat[te_i])
                for loc_i, glb_i in enumerate(te_i):
                    preds_setting[int(glb_i)] = p_fold[loc_i]
            m_setting = compute_classification_metrics(y_all, preds_setting, classes=FAULT_CLASSES)
            leakage_experiment[label_setting] = {
                "accuracy": m_setting["accuracy"],
                "macro_f1": m_setting["macro_f1"],
                "balanced_accuracy": m_setting["balanced_accuracy"],
            }

    return {
        "models": results,
        "leakage_comparison": leakage_experiment,
        "n_splits": len(group_folds),
        "n_windows": len(df),
        "n_runs": int(df["run_id"].nunique()),
        "source_types": sorted(df["source_type"].unique().tolist()),
    }
