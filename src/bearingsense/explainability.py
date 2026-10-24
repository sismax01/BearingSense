# =============================================================================
# BearingSense — Explainability, Out-of-Fold Importance & Recording Aggregation
# Engineered by : SISMAX (sismax01)
# =============================================================================
"""Explainability and physical diagnostic evidence attribution for BearingSense.

Provides three levels of explainability for model predictions:
  1. **Out-of-Fold Global Permutation Importance**:
     - Computes feature importance strictly on held-out out-of-fold validation splits
       (preventing in-sample overfitting bias) alongside tree Gini impurity reduction.
  2. **Local Instance Attribution & Physical Evidence Mapping**:
     - Computes per-feature marginal probability contributions (`probability_contribution`)
       by replacing each feature in the inspected window with healthy ('Normal') baseline
       expectation and measuring the drop in predicted probability.
  3. **Full-Recording Multi-Window Diagnostic Aggregation**:
     - Slices an entire continuous recording into windows, classifies all windows,
       computes recording-level class vote distribution & mean probability, and selects
       the highest-confidence window for detailed local inspection.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from bearingsense.bearing_kinematics import (
    BearingSpec,
    SKF_6205_DRIVE_END,
    compute_fault_frequencies,
)
from bearingsense.evaluation import create_stratified_group_folds
from bearingsense.features import (
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    extract_features_from_window,
    slice_signal_windows,
)
from bearingsense.modeling import BaseBearingClassifier, RandomForestBearingClassifier


def _multiclass_brier_loss(probs: np.ndarray, y_true: np.ndarray, classes: Sequence[str]) -> float:
    """Compute multiclass Brier loss (mean squared probability error) across samples."""
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    Y_oh = np.zeros_like(probs, dtype=np.float64)
    for row_i, lbl in enumerate(y_true):
        if lbl in cls_to_idx:
            Y_oh[row_i, cls_to_idx[lbl]] = 1.0
    return float(np.mean(np.sum((probs - Y_oh) ** 2, axis=1)))


def compute_permutation_importance(
    model: BaseBearingClassifier,
    X_val: np.ndarray,
    y_val: Sequence[str],
    feature_names: List[str] = FEATURE_NAMES,
    n_repeats: int = 8,
    seed: int = 42,
    df_for_oof: Optional[pd.DataFrame] = None,
    n_splits: int = 4,
) -> List[Dict]:
    """Compute model-agnostic permutation importance.

    SISMAX Methodological Note:
    When `df_for_oof` (containing `run_id` and `label`) is provided, permutation importance
    is evaluated strictly out-of-fold across leak-free Stratified Group K-Folds so that
    feature importance reflects true cross-run generalization rather than in-sample memorization.
    """
    X_arr = np.asarray(X_val, dtype=np.float64)
    y_arr = np.asarray(y_val, dtype=str)
    rng = np.random.default_rng(seed)

    gini_imp = getattr(model, "feature_importances_", None)

    if df_for_oof is not None and "run_id" in df_for_oof.columns:
        folds = create_stratified_group_folds(df_for_oof, n_splits=n_splits, seed=seed)
        fold_models: List[Tuple[BaseBearingClassifier, np.ndarray, np.ndarray]] = []
        for tr_idx, te_idx in folds:
            m_fold = RandomForestBearingClassifier(n_estimators=50, max_depth=6, seed=seed)
            m_fold.fit(X_arr[tr_idx], y_arr[tr_idx], feature_names=feature_names)
            fold_models.append((m_fold, X_arr[te_idx], y_arr[te_idx]))

        importances: List[Dict] = []
        for f_idx, f_name in enumerate(feature_names):
            increases: List[float] = []
            for _ in range(n_repeats):
                fold_losses: List[float] = []
                for m_fold, X_te, y_te in fold_models:
                    base_p = m_fold.predict_proba(X_te)
                    base_l = _multiclass_brier_loss(base_p, y_te, m_fold.classes_)
                    X_perm = X_te.copy()
                    X_perm[:, f_idx] = rng.permutation(X_perm[:, f_idx])
                    perm_p = m_fold.predict_proba(X_perm)
                    perm_l = _multiclass_brier_loss(perm_p, y_te, m_fold.classes_)
                    fold_losses.append(max(0.0, perm_l - base_l))
                increases.append(float(np.mean(fold_losses)))

            mean_inc = float(np.mean(increases))
            std_inc = float(np.std(increases))
            intrinsic_val = float(gini_imp[f_idx]) if gini_imp is not None and f_idx < len(gini_imp) else 0.0
            combined = 0.65 * mean_inc + 0.35 * intrinsic_val
            importances.append({
                "feature": f_name,
                "description": FEATURE_DESCRIPTIONS.get(f_name, ""),
                "importance_mean": combined,
                "importance_std": std_inc,
                "permutation_brier_increase": mean_inc,
                "gini_importance": intrinsic_val,
                "evaluation_mode": "out_of_fold_group_cv",
            })

        importances.sort(key=lambda d: d["importance_mean"], reverse=True)
        return importances

    # Standard held-out validation array evaluation
    base_probs = model.predict_proba(X_arr)
    base_loss = _multiclass_brier_loss(base_probs, y_arr, model.classes_)

    importances = []
    for f_idx, f_name in enumerate(feature_names):
        increases = []
        for _ in range(n_repeats):
            X_perm = X_arr.copy()
            X_perm[:, f_idx] = rng.permutation(X_perm[:, f_idx])
            perm_probs = model.predict_proba(X_perm)
            perm_loss = _multiclass_brier_loss(perm_probs, y_arr, model.classes_)
            increases.append(max(0.0, perm_loss - base_loss))

        mean_inc = float(np.mean(increases))
        std_inc = float(np.std(increases))
        intrinsic_val = float(gini_imp[f_idx]) if gini_imp is not None and f_idx < len(gini_imp) else 0.0
        combined = 0.65 * mean_inc + 0.35 * intrinsic_val

        importances.append({
            "feature": f_name,
            "description": FEATURE_DESCRIPTIONS.get(f_name, ""),
            "importance_mean": combined,
            "importance_std": std_inc,
            "permutation_brier_increase": mean_inc,
            "gini_importance": intrinsic_val,
            "evaluation_mode": "validation_split",
        })

    importances.sort(key=lambda d: d["importance_mean"], reverse=True)
    return importances


def explain_single_prediction(
    model: BaseBearingClassifier,
    feature_vector: Dict[str, float] | np.ndarray,
    normal_baseline_mean: np.ndarray,
    normal_baseline_std: np.ndarray,
    rpm: float = 1797.0,
    bearing: BearingSpec = SKF_6205_DRIVE_END,
    feature_names: List[str] = FEATURE_NAMES,
    top_k: int = 6,
) -> Dict:
    """Generate a local explanation and physical evidence report for a single vibration window."""
    if isinstance(feature_vector, dict):
        x_row = np.array([float(feature_vector[k]) for k in feature_names], dtype=np.float64)
        feat_dict = {k: float(feature_vector[k]) for k in feature_names}
    else:
        x_row = np.asarray(feature_vector, dtype=np.float64).ravel()
        feat_dict = {k: float(x_row[i]) for i, k in enumerate(feature_names)}

    probs = model.predict_proba(x_row.reshape(1, -1))[0]
    pred_idx = int(np.argmax(probs))
    pred_class = model.classes_[pred_idx]
    pred_confidence = float(probs[pred_idx])

    class_probabilities = {
        cls_name: float(probs[i]) for i, cls_name in enumerate(model.classes_)
    }

    attributions: List[Dict] = []
    for i, f_name in enumerate(feature_names):
        x_cf = x_row.copy()
        x_cf[i] = float(normal_baseline_mean[i])
        cf_probs = model.predict_proba(x_cf.reshape(1, -1))[0]
        delta_prob = float(probs[pred_idx] - cf_probs[pred_idx])
        z_score = float((x_row[i] - normal_baseline_mean[i]) / max(normal_baseline_std[i], 1e-9))

        attributions.append({
            "feature": f_name,
            "description": FEATURE_DESCRIPTIONS.get(f_name, f_name),
            "observed_value": float(x_row[i]),
            "healthy_baseline_mean": float(normal_baseline_mean[i]),
            "z_score_vs_healthy": z_score,
            "probability_contribution": delta_prob,
        })

    attributions.sort(
        key=lambda d: (abs(d["probability_contribution"]), abs(d["z_score_vs_healthy"])),
        reverse=True,
    )
    top_drivers = attributions[:top_k]

    fault_freqs = compute_fault_frequencies(rpm=rpm, bearing=bearing)
    dom_freq = feat_dict.get("env_dominant_freq_hz", 0.0)

    physical_rationale = _build_physical_narrative(
        pred_class=pred_class,
        pred_confidence=pred_confidence,
        feat_dict=feat_dict,
        fault_freqs=fault_freqs,
        dom_freq=dom_freq,
    )

    return {
        "predicted_class": pred_class,
        "confidence": pred_confidence,
        "class_probabilities": class_probabilities,
        "fault_frequencies_hz": fault_freqs,
        "top_feature_attributions": top_drivers,
        "all_feature_attributions": attributions,
        "physical_narrative": physical_rationale,
    }


def analyze_full_recording(
    signal_1d: np.ndarray,
    fs: float,
    rpm: float,
    model: BaseBearingClassifier,
    normal_baseline_mean: np.ndarray,
    normal_baseline_std: np.ndarray,
    window_size: int = 2048,
    step_size: int = 1024,
    bearing: BearingSpec = SKF_6205_DRIVE_END,
) -> Tuple[Dict, np.ndarray]:
    """Slice a full continuous recording into windows, aggregate predictions, and explain.

    SISMAX Multi-Window Diagnostic Aggregation:
    Instead of inspecting only the first 2048 samples of a recording, this function evaluates
    every sliding window across the entire recording, computes recording-level majority vote,
    window consistency percentage, and mean class probabilities, and returns the detailed
    local explanation for the representative window of the diagnosed class.
    """
    sig = np.asarray(signal_1d, dtype=np.float64)
    if sig.size < window_size:
        windows = [sig]
    else:
        windows = slice_signal_windows(sig, window_size=window_size, step_size=step_size)

    feat_dicts = [
        extract_features_from_window(w, fs=fs, rpm=rpm, bearing=bearing) for w in windows
    ]
    X_wins = np.array(
        [[fd[k] for k in FEATURE_NAMES] for fd in feat_dicts], dtype=np.float64
    )
    probs_matrix = model.predict_proba(X_wins)
    win_pred_indices = np.argmax(probs_matrix, axis=1)
    win_preds = [model.classes_[int(i)] for i in win_pred_indices]

    mean_probs = np.mean(probs_matrix, axis=0)
    overall_idx = int(np.argmax(mean_probs))
    overall_class = model.classes_[overall_idx]
    overall_confidence = float(mean_probs[overall_idx])

    vote_counts = {c: int( sum(1 for p in win_preds if p == c) ) for c in model.classes_}
    consistency_ratio = float(vote_counts[overall_class] / len(windows))

    # Select the window with highest predicted probability for the overall diagnosed class
    matching_win_idx = int(np.argmax(probs_matrix[:, overall_idx]))
    rep_window = windows[matching_win_idx]
    rep_feats = feat_dicts[matching_win_idx]

    explanation = explain_single_prediction(
        model=model,
        feature_vector=rep_feats,
        normal_baseline_mean=normal_baseline_mean,
        normal_baseline_std=normal_baseline_std,
        rpm=rpm,
        bearing=bearing,
    )
    explanation["recording_summary"] = {
        "total_windows_evaluated": len(windows),
        "overall_predicted_class": overall_class,
        "mean_class_confidence": overall_confidence,
        "window_agreement_ratio": consistency_ratio,
        "window_vote_counts": vote_counts,
        "representative_window_index": matching_win_idx,
    }
    return explanation, rep_window


def _build_physical_narrative(
    pred_class: str,
    pred_confidence: float,
    feat_dict: Dict[str, float],
    fault_freqs: Dict[str, float],
    dom_freq: float,
) -> str:
    """Construct a concise engineering interpretation of the spectral and statistical evidence."""
    bpfo_hz = fault_freqs["BPFO"]
    bpfi_hz = fault_freqs["BPFI"]
    bsf_hz = fault_freqs["BSF"]
    bdf_hz = fault_freqs["BDF"]
    fr_hz = fault_freqs["fr"]

    bpfo_e = feat_dict.get("env_bpfo_energy_ratio", 0.0) * 100.0
    bpfi_e = feat_dict.get("env_bpfi_energy_ratio", 0.0) * 100.0
    bsf_e = feat_dict.get("env_bsf_energy_ratio", 0.0) * 100.0
    sb_e = feat_dict.get("env_ftf_sideband_ratio", 0.0) * 100.0
    kurt = feat_dict.get("td_kurtosis", 0.0)
    rms = feat_dict.get("td_rms", 0.0)

    if pred_class == "Normal":
        return (
            f"Classified as Normal ({pred_confidence:.1%} confidence). "
            f"Overall RMS vibration ({rms:.3f}) and time-domain kurtosis ({kurt:.2f}) remain low, "
            f"and the demodulated Hilbert envelope spectrum shows minimal harmonic concentration at "
            f"BPFO ({bpfo_hz:.1f} Hz: {bpfo_e:.1f}%), BPFI ({bpfi_hz:.1f} Hz: {bpfi_e:.1f}%), "
            f"or BSF ({bsf_hz:.1f} Hz: {bsf_e:.1f}%)."
        )
    if pred_class == "Outer_Race_Fault":
        return (
            f"Classified as Outer_Race_Fault ({pred_confidence:.1%} confidence). "
            f"The Hilbert envelope spectrum exhibits strong harmonic energy concentration ({bpfo_e:.1f}% of band energy) "
            f"aligned with the Ball Pass Frequency Outer race (BPFO = {bpfo_hz:.1f} Hz, 2x = {2*bpfo_hz:.1f} Hz), "
            f"with dominant envelope peak at {dom_freq:.1f} Hz and elevated waveform kurtosis ({kurt:.2f})."
        )
    if pred_class == "Inner_Race_Fault":
        return (
            f"Classified as Inner_Race_Fault ({pred_confidence:.1%} confidence). "
            f"Significant envelope energy ({bpfi_e:.1f}%) is concentrated at the Ball Pass Frequency Inner race "
            f"(BPFI = {bpfi_hz:.1f} Hz) accompanied by shaft-rate modulation sidebands (BPFI ± fr at "
            f"{bpfi_hz - fr_hz:.1f} / {bpfi_hz + fr_hz:.1f} Hz, sideband energy = {sb_e:.1f}%) caused by "
            f"the defect rotating through the radial load zone."
        )
    if pred_class == "Ball_Fault":
        return (
            f"Classified as Ball_Fault ({pred_confidence:.1%} confidence). "
            f"Envelope energy ({bsf_e:.1f}%) is concentrated around the Ball Spin Frequency harmonics "
            f"(BSF = {bsf_hz:.1f} Hz, 2xBSF = {bdf_hz:.1f} Hz) alongside cage-frequency modulation "
            f"(FTF = {fault_freqs['FTF']:.1f} Hz, sideband ratio = {sb_e:.1f}%), characteristic of "
            f"a rolling element defect striking both inner and outer raceways."
        )
    return f"Classified as {pred_class} ({pred_confidence:.1%} confidence)."
