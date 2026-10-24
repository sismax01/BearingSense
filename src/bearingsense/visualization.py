# =============================================================================
# BearingSense — Diagnostic Visualization & HTML Dashboard Builder
# Engineered by : SISMAX (sismax01)
# =============================================================================
"""Diagnostic visualization and interactive HTML report generation for BearingSense.

Generates publication-grade Matplotlib figures and a self-contained HTML engineering
dashboard with embedded base64 plots, model comparison tables, leak-free vs. leaky split
diagnostics, and local feature attribution explanations.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, List, Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bearingsense.bearing_kinematics import (
    BearingSpec,
    SKF_6205_DRIVE_END,
    compute_fault_frequencies,
)
from bearingsense.signal_processing import (
    compute_envelope_spectrum,
    compute_power_spectrum,
)
from bearingsense.simulation import FAULT_CLASSES

PALETTE = {
    "Normal": "#1b9e77",
    "Outer_Race_Fault": "#d95f02",
    "Inner_Race_Fault": "#7570b3",
    "Ball_Fault": "#e7298a",
    "navy": "#1e3a5f",
    "slate": "#4a5568",
    "accent": "#2b6cb0",
    "warning": "#c53030",
}


def plot_envelope_diagnostics_grid(
    sample_records: Sequence[Dict],
    output_path: str | Path,
    bearing: BearingSpec = SKF_6205_DRIVE_END,
) -> Path:
    """Plot a 4x2 diagnostic grid comparing time-domain waveforms & Hilbert envelope spectra across classes."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    by_class: Dict[str, Dict] = {}
    for rec in sample_records:
        lbl = str(rec["label"])
        if lbl not in by_class:
            by_class[lbl] = rec

    ordered_classes = [c for c in FAULT_CLASSES if c in by_class]
    n_rows = len(ordered_classes)

    fig, axes = plt.subplots(n_rows, 2, figsize=(13.5, 2.85 * n_rows), dpi=160)
    if n_rows == 1:
        axes = np.array([axes])

    is_synthetic = any("synthetic" in str(r.get("source_type", "")).lower() for r in sample_records)
    source_tag = "[SYNTHETIC SIMULATION DATA]" if is_synthetic else "[EMPIRICAL CWRU DATA]"

    fig.suptitle(
        f"BearingSense — Time-Domain Waveform & Hilbert Envelope Spectrum Comparison {source_tag}",
        fontsize=13,
        fontweight="bold",
        color="#1a202c",
        y=0.99,
    )

    for row_idx, cls_name in enumerate(ordered_classes):
        rec = by_class[cls_name]
        sig = np.asarray(rec["signal"], dtype=np.float64)
        fs = float(rec.get("fs", 12000.0))
        rpm = float(rec.get("rpm", 1797.0))
        color = PALETTE.get(cls_name, "#2b6cb0")

        n_show = min(int(0.12 * fs), sig.size)
        t_ms = (np.arange(n_show) / fs) * 1000.0
        sig_win = sig[: min(4096, sig.size)]

        env_freqs, env_amps, env_time = compute_envelope_spectrum(
            sig_win, fs=fs, bandpass_range=(2000.0, 5000.0), max_freq_hz=400.0
        )
        fault_freqs = compute_fault_frequencies(rpm=rpm, bearing=bearing)

        ax_time = axes[row_idx, 0]
        ax_env = axes[row_idx, 1]

        ax_time.plot(t_ms, sig[:n_show], color=color, lw=0.85, alpha=0.85, label="Raw Vibration")
        ax_time.plot(
            t_ms,
            env_time[:n_show],
            color="#1a202c",
            lw=1.15,
            linestyle="--",
            alpha=0.85,
            label="Resonance Envelope |H(x)|",
        )
        ax_time.set_title(f"{cls_name.replace('_', ' ')} — Time Waveform ({int(rpm)} RPM)", fontsize=10, fontweight="bold")
        ax_time.set_xlabel("Time (ms)", fontsize=9)
        ax_time.set_ylabel("Accel. (g)", fontsize=9)
        ax_time.grid(True, linestyle=":", alpha=0.5)
        if row_idx == 0:
            ax_time.legend(loc="upper right", fontsize=8, framealpha=0.9)

        ax_env.plot(env_freqs, env_amps, color=color, lw=1.15)
        ax_env.set_xlim(0, 385)
        y_max = float(np.max(env_amps)) if env_amps.size > 0 else 1.0
        ax_env.set_ylim(0, y_max * 1.18)
        ax_env.set_title(
            f"{cls_name.replace('_', ' ')} — Demodulated Hilbert Envelope Spectrum",
            fontsize=10,
            fontweight="bold",
        )
        ax_env.set_xlabel("Frequency (Hz)", fontsize=9)
        ax_env.set_ylabel("Envelope Amp.", fontsize=9)
        ax_env.grid(True, linestyle=":", alpha=0.5)

        markers = [
            ("1x BPFO", fault_freqs["BPFO"], "#d95f02", 0.96),
            ("2x BSF", fault_freqs["BDF"], "#e7298a", 0.96),
            ("1x BPFI", fault_freqs["BPFI"], "#7570b3", 0.72),
            ("2x BPFO", 2.0 * fault_freqs["BPFO"], "#d95f02", 0.96),
            ("2x BPFI", 2.0 * fault_freqs["BPFI"], "#7570b3", 0.96),
        ]
        for m_label, m_freq, m_col, y_frac in markers:
            if m_freq <= 375:
                ax_env.axvline(m_freq, color=m_col, linestyle="--", lw=0.95, alpha=0.72)
                ax_env.text(
                    m_freq + 2.5,
                    y_max * 1.18 * y_frac,
                    f"{m_label}\n({m_freq:.0f}Hz)",
                    color=m_col,
                    fontsize=7,
                    fontweight="bold",
                    va="top",
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
                )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_single_window_inspection(
    window: np.ndarray,
    fs: float,
    rpm: float,
    explanation: Dict,
    output_path: str | Path,
    source_type: str = "synthetic_simulation",
    bearing: BearingSpec = SKF_6205_DRIVE_END,
) -> Path:
    """Create a 4-panel explainable diagnostic figure for a single inspected vibration window."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.3), dpi=160)
    pred_cls = explanation["predicted_class"]
    conf = explanation["confidence"]
    tag = "SYNTHETIC DEMO" if "synthetic" in source_type.lower() else "EMPIRICAL RECORDING"

    fig.suptitle(
        f"BearingSense Explainable Diagnostic Report — Predicted: {pred_cls.replace('_', ' ')} "
        f"({conf:.1%} Confidence) [{tag}]",
        fontsize=12.5,
        fontweight="bold",
        color="#1a202c",
        y=0.98,
    )

    ax1 = axes[0, 0]
    n_show = min(int(0.12 * fs), window.size)
    t_ms = (np.arange(n_show) / fs) * 1000.0
    env_freqs, env_amps, env_sig = compute_envelope_spectrum(window, fs=fs, max_freq_hz=400.0)
    cls_color = PALETTE.get(pred_cls, "#2b6cb0")

    ax1.plot(t_ms, window[:n_show], color=cls_color, lw=0.85, alpha=0.8, label="Vibration Waveform")
    ax1.plot(t_ms, env_sig[:n_show], color="#1a202c", lw=1.2, linestyle="--", label="Hilbert Envelope |H(x)|")
    ax1.set_title("1. Time-Domain Vibration & Bandpass Resonance Envelope", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Time (ms)", fontsize=9)
    ax1.set_ylabel("Acceleration (g)", fontsize=9)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = axes[0, 1]
    fft_freqs, fft_amps = compute_power_spectrum(window, fs=fs)
    ax2.plot(fft_freqs, fft_amps, color="#4a5568", lw=0.8)
    ax2.axvspan(2000, min(5000, 0.48 * fs), color="#ebf8ff", alpha=0.7, label="Demodulation Band (2–5 kHz)")
    ax2.set_title("2. Broadband FFT Spectrum & Structural Resonance Band", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Frequency (Hz)", fontsize=9)
    ax2.set_ylabel("Amplitude", fontsize=9)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.5)

    ax3 = axes[1, 0]
    ax3.plot(env_freqs, env_amps, color=cls_color, lw=1.2)
    fault_freqs = compute_fault_frequencies(rpm=rpm, bearing=bearing)
    y_max = float(np.max(env_amps)) if env_amps.size > 0 else 1.0
    ax3.set_ylim(0, y_max * 1.18)

    for lbl, f_val, col, y_frac in [
        ("1x BPFO", fault_freqs["BPFO"], "#d95f02", 0.96),
        ("2x BSF", fault_freqs["BDF"], "#e7298a", 0.96),
        ("1x BPFI", fault_freqs["BPFI"], "#7570b3", 0.72),
        ("2x BPFO", 2.0 * fault_freqs["BPFO"], "#d95f02", 0.96),
        ("2x BPFI", 2.0 * fault_freqs["BPFI"], "#7570b3", 0.96),
    ]:
        if f_val <= 380:
            ax3.axvline(f_val, color=col, linestyle="--", lw=1.0, alpha=0.8)
            ax3.text(
                f_val + 2.5,
                y_max * 1.18 * y_frac,
                f"{lbl}\n({f_val:.0f}Hz)",
                color=col,
                fontsize=7.5,
                fontweight="bold",
                va="top",
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            )

    ax3.set_xlim(0, 390)
    ax3.set_title(f"3. Hilbert Envelope Spectrum vs. Kinematic Fault Orders ({int(rpm)} RPM)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Frequency (Hz)", fontsize=9)
    ax3.set_ylabel("Demodulated Amplitude", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.5)

    ax4 = axes[1, 1]
    top_attrs = list(reversed(explanation["top_feature_attributions"][:6]))
    f_labels = [a["feature"] for a in top_attrs]
    f_vals = [a["probability_contribution"] for a in top_attrs]
    bar_colors = ["#2b6cb0" if v >= 0 else "#c53030" for v in f_vals]

    y_pos = np.arange(len(f_labels))
    ax4.barh(y_pos, f_vals, color=bar_colors, height=0.6, alpha=0.88)
    ax4.axvline(0.0, color="#1a202c", lw=0.9)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(f_labels, fontsize=8.5)
    ax4.set_title("4. Local Feature Attribution (Marginal Probability Contribution ΔP)", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Contribution to Predicted Class Probability (ΔP vs. Healthy Baseline)", fontsize=8.5)
    ax4.grid(True, axis="x", linestyle=":", alpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_evaluation_summary(
    cv_results: Dict,
    perm_importances: List[Dict],
    output_path: str | Path,
) -> Path:
    """Plot a 3-panel evaluation summary: Confusion Matrix, Model Benchmark, and Out-of-Fold Feature Importance."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2), dpi=160)
    is_syn = any("synthetic" in s.lower() for s in cv_results.get("source_types", []))
    tag = "SYNTHETIC VERIFICATION" if is_syn else "EMPIRICAL CWRU EVALUATION"

    fig.suptitle(
        f"BearingSense — Leak-Free Stratified Group K-Fold Evaluation ({cv_results['n_runs']} Runs, {cv_results['n_windows']} Windows) [{tag}]",
        fontsize=12.5,
        fontweight="bold",
        color="#1a202c",
        y=1.01,
    )

    ax_cm = axes[0]
    rf_metrics = cv_results["models"]["Random Forest"]
    cm = np.array(rf_metrics["confusion_matrix"], dtype=int)
    classes = [c.replace("_Fault", "").replace("_", " ") for c in rf_metrics["classes"]]

    ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks(np.arange(len(classes)))
    ax_cm.set_yticks(np.arange(len(classes)))
    ax_cm.set_xticklabels(classes, fontsize=8.5, rotation=20)
    ax_cm.set_yticklabels(classes, fontsize=8.5)
    ax_cm.set_xlabel("Predicted Condition", fontsize=9.5, fontweight="bold")
    ax_cm.set_ylabel("True Condition", fontsize=9.5, fontweight="bold")
    ax_cm.set_title(
        f"Random Forest Confusion Matrix\n(Run-Grouped Macro-F1 = {rf_metrics['macro_f1']:.3f})",
        fontsize=10,
        fontweight="bold",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "#1a202c",
                fontweight="bold",
                fontsize=9.5,
            )

    ax_mod = axes[1]
    model_names = list(cv_results["models"].keys())
    short_label_map = {
        "Physics Rule Baseline": "Physics\nRule",
        "Logistic Regression (L2)": "Logistic\nReg. (L2)",
        "Gaussian Discriminant": "Gaussian\nDiscrim.",
        "Random Forest": "Random\nForest",
    }
    short_labels = [short_label_map.get(m, m) for m in model_names]
    macro_f1s = [cv_results["models"][m]["macro_f1"] for m in model_names]
    bal_accs = [cv_results["models"][m]["balanced_accuracy"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35
    b1 = ax_mod.bar(x - width / 2, macro_f1s, width, label="Macro F1-Score", color="#2b6cb0")
    ax_mod.bar(x + width / 2, bal_accs, width, label="Balanced Accuracy", color="#38b2ac")

    ax_mod.set_ylim(0.0, 1.13)
    ax_mod.set_xticks(x)
    ax_mod.set_xticklabels(short_labels, fontsize=8.5)
    ax_mod.set_ylabel("Score (0.0 – 1.0)", fontsize=9.5)
    ax_mod.set_title("Model Benchmark (Leak-Free Group CV)", fontsize=10, fontweight="bold")
    ax_mod.legend(loc="lower right", fontsize=8.5)
    ax_mod.grid(True, axis="y", linestyle=":", alpha=0.5)

    for bar in b1:
        h = bar.get_height()
        ax_mod.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", fontsize=8, fontweight="bold")

    ax_imp = axes[2]
    top_p = list(reversed(perm_importances[:8]))
    p_names = [d["feature"] for d in top_p]
    p_means = [d["importance_mean"] for d in top_p]
    p_stds = [d["importance_std"] for d in top_p]

    y_p = np.arange(len(p_names))
    ax_imp.barh(y_p, p_means, xerr=p_stds, color="#4c51bf", alpha=0.85, height=0.6, capsize=3)
    ax_imp.set_yticks(y_p)
    ax_imp.set_yticklabels(p_names, fontsize=8.5)
    ax_imp.set_xlabel("Out-of-Fold Permutation & Impurity Importance", fontsize=9)
    ax_imp.set_title("Global Feature Importance (Top 8)", fontsize=10, fontweight="bold")
    ax_imp.grid(True, axis="x", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def _encode_png_base64(png_path: Path) -> str:
    return base64.b64encode(png_path.read_bytes()).decode("ascii")


def generate_html_diagnostic_dashboard(
    cv_results: Dict,
    perm_importances: List[Dict],
    sample_explanation: Dict,
    envelope_grid_png: Path,
    eval_summary_png: Path,
    single_inspect_png: Path,
    output_html_path: str | Path,
) -> Path:
    """Generate a standalone, self-contained HTML engineering dashboard with embedded base64 figures."""
    out_file = Path(output_html_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    is_synthetic = any("synthetic" in s.lower() for s in cv_results.get("source_types", []))
    banner_html = ""
    if is_synthetic:
        banner_html = """
        <div class="alert-banner">
            <strong>SCIENTIFIC INTEGRITY NOTICE — SYNTHETIC DEMONSTRATION DATA:</strong>
            The metrics and figures in this report were generated using BearingSense's physics-informed
            synthetic bearing vibration simulator (<code>source_type='synthetic_simulation'</code>) for
            reproducible software verification. Synthetic results must not be cited as empirical real-world
            CWRU benchmark performance. To run this exact pipeline on real experimental data, download the
            MATLAB <code>.mat</code> files from the Case Western Reserve University Bearing Data Center and
            run <code>bearingsense evaluate --data-dir data/raw</code>.
        </div>
        """

    img_env_b64 = _encode_png_base64(envelope_grid_png)
    img_eval_b64 = _encode_png_base64(eval_summary_png)
    img_insp_b64 = _encode_png_base64(single_inspect_png)

    model_rows_html = ""
    for m_name, m_data in cv_results["models"].items():
        brier_str = f"{m_data['brier_score']:.4f}" if m_data.get("brier_score") is not None else "N/A"
        model_rows_html += f"""
        <tr>
            <td><strong>{m_name}</strong></td>
            <td>{m_data['accuracy']:.3f}</td>
            <td>{m_data['balanced_accuracy']:.3f}</td>
            <td>{m_data['macro_precision']:.3f}</td>
            <td>{m_data['macro_recall']:.3f}</td>
            <td><strong>{m_data['macro_f1']:.3f}</strong> (±{m_data.get('fold_macro_f1_std', 0.0):.3f})</td>
            <td>{brier_str}</td>
        </tr>
        """

    rf_per_class_rows = ""
    for c_name, c_metrics in cv_results["models"]["Random Forest"]["per_class"].items():
        rf_per_class_rows += f"""
        <tr>
            <td><strong>{c_name.replace('_', ' ')}</strong></td>
            <td>{c_metrics['precision']:.3f}</td>
            <td>{c_metrics['recall']:.3f}</td>
            <td>{c_metrics['specificity']:.3f}</td>
            <td><strong>{c_metrics['f1_score']:.3f}</strong></td>
            <td>{int(c_metrics['support'])}</td>
        </tr>
        """

    leak_rows_html = ""
    for setting_name, s_metrics in cv_results.get("leakage_comparison", {}).items():
        leak_rows_html += f"""
        <tr>
            <td>{setting_name}</td>
            <td>{s_metrics['accuracy']:.3f}</td>
            <td>{s_metrics['balanced_accuracy']:.3f}</td>
            <td><strong>{s_metrics['macro_f1']:.3f}</strong></td>
        </tr>
        """

    attr_rows_html = ""
    for attr in sample_explanation["top_feature_attributions"]:
        sign = "+" if attr["probability_contribution"] >= 0 else ""
        attr_rows_html += f"""
        <tr>
            <td><code>{attr['feature']}</code></td>
            <td>{attr['description']}</td>
            <td>{attr['observed_value']:.4f}</td>
            <td>{attr['healthy_baseline_mean']:.4f}</td>
            <td>{attr['z_score_vs_healthy']:+.2f}σ</td>
            <td><strong>{sign}{attr['probability_contribution']:.3f}</strong></td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BearingSense — Explainable Bearing Fault Diagnostics Report (by SISMAX)</title>
    <style>
        :root {{
            --navy: #1e3a5f;
            --blue: #2b6cb0;
            --bg: #f7fafc;
            --card: #ffffff;
            --text: #1a202c;
            --muted: #4a5568;
            --border: #e2e8f0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.55;
            margin: 0;
            padding: 0;
        }}
        header {{
            background: linear-gradient(135deg, #1e3a5f 0%, #2b6cb0 100%);
            color: white;
            padding: 2rem 2.5rem;
        }}
        header h1 {{
            margin: 0 0 0.4rem 0;
            font-size: 1.85rem;
        }}
        header p {{
            margin: 0;
            opacity: 0.92;
            font-size: 1rem;
        }}
        .container {{
            max-width: 1240px;
            margin: 1.5rem auto;
            padding: 0 1.5rem;
        }}
        .alert-banner {{
            background-color: #fffaf0;
            border-left: 5px solid #dd6b20;
            color: #7b341e;
            padding: 1rem 1.25rem;
            border-radius: 6px;
            margin-bottom: 1.5rem;
            font-size: 0.94rem;
        }}
        .card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}
        h2 {{
            color: var(--navy);
            margin-top: 0;
            border-bottom: 2px solid var(--border);
            padding-bottom: 0.5rem;
            font-size: 1.3rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.92rem;
        }}
        th, td {{
            padding: 0.65rem 0.85rem;
            border-bottom: 1px solid var(--border);
            text-align: left;
        }}
        th {{
            background-color: #edf2f7;
            color: var(--navy);
            font-weight: 600;
        }}
        .figure-box {{
            text-align: center;
            margin: 1.2rem 0;
        }}
        .figure-box img {{
            max-width: 100%;
            height: auto;
            border: 1px solid var(--border);
            border-radius: 6px;
        }}
        .narrative-box {{
            background-color: #ebf8ff;
            border-left: 4px solid var(--blue);
            padding: 1rem 1.2rem;
            border-radius: 4px;
            margin: 1rem 0;
            font-size: 0.96rem;
        }}
        footer {{
            text-align: center;
            padding: 1.5rem;
            color: var(--muted);
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <header>
        <h1>BearingSense — Explainable Bearing Fault Diagnostics</h1>
        <p>Hilbert Envelope Spectrum Demodulation, Leak-Free Run-Grouped Evaluation &amp; Physical Evidence Attribution &bull; Engineered by SISMAX</p>
    </header>
    <div class="container">
        {banner_html}

        <div class="card">
            <h2>1. Leak-Free Stratified Group K-Fold Model Evaluation ({cv_results['n_runs']} Runs, {cv_results['n_windows']} Windows)</h2>
            <p>All windows extracted from the same physical operating run (<code>run_id</code>) are strictly isolated into either training or testing folds to prevent window-level autocorrelation leakage.</p>
            <table>
                <thead>
                    <tr>
                        <th>Model Architecture</th>
                        <th>Accuracy</th>
                        <th>Balanced Accuracy</th>
                        <th>Macro Precision</th>
                        <th>Macro Recall</th>
                        <th>Macro F1-Score (±SD)</th>
                        <th>Multiclass Brier Score</th>
                    </tr>
                </thead>
                <tbody>
                    {model_rows_html}
                </tbody>
            </table>

            <h3>Per-Class Diagnostic Performance (Random Forest Classifier)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Bearing Condition Class</th>
                        <th>Precision</th>
                        <th>Recall (Sensitivity)</th>
                        <th>Specificity</th>
                        <th>F1-Score</th>
                        <th>Window Support</th>
                    </tr>
                </thead>
                <tbody>
                    {rf_per_class_rows}
                </tbody>
            </table>

            <div class="figure-box">
                <img src="data:image/png;base64,{img_eval_b64}" alt="Evaluation Summary Plot">
            </div>
        </div>

        <div class="card">
            <h2>2. Methodological Study: Why Naive Random Window Splitting Causes Data Leakage</h2>
            <p>In vibration monitoring, adjacent time windows from the same continuous acquisition share identical shaft speed, load, background noise floor, and sensor mounting resonance. Splitting windows randomly across train and test sets allows a classifier to memorize run-specific spectral signatures rather than generalizing across operating runs:</p>
            <table>
                <thead>
                    <tr>
                        <th>Evaluation Split Strategy &amp; Feature Set</th>
                        <th>Accuracy</th>
                        <th>Balanced Accuracy</th>
                        <th>Macro F1-Score</th>
                    </tr>
                </thead>
                <tbody>
                    {leak_rows_html}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>3. Physical Signal Processing: Time Domain vs. Demodulated Hilbert Envelope Spectrum</h2>
            <p>Localized defects on the outer raceway (BPFO), inner raceway (BPFI), or rolling elements (BSF / 2&times;BSF) produce periodic mechanical impacts that excite high-frequency structural resonances (2–5 kHz). Bandpass filtering followed by the Hilbert transform extracts the repetition rate of these impacts:</p>
            <div class="figure-box">
                <img src="data:image/png;base64,{img_env_b64}" alt="Envelope Spectrum Diagnostics Grid">
            </div>
        </div>

        <div class="card">
            <h2>4. Explainable Single-Window Diagnostic Inspection</h2>
            <div class="narrative-box">
                <strong>Physical Diagnostic Synthesis:</strong> {sample_explanation['physical_narrative']}
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Feature Identifier</th>
                        <th>Physical Meaning</th>
                        <th>Observed Value</th>
                        <th>Healthy Mean</th>
                        <th>Z-Score vs. Healthy</th>
                        <th>Marginal Contribution (&Delta;P)</th>
                    </tr>
                </thead>
                <tbody>
                    {attr_rows_html}
                </tbody>
            </table>
            <div class="figure-box">
                <img src="data:image/png;base64,{img_insp_b64}" alt="Single Window Inspection Plot">
            </div>
        </div>
    </div>
    <footer>
        BearingSense v0.1.0 &bull; Engineered by <strong>SISMAX (sismax01)</strong> &bull; Educational &amp; Research Portfolio Project &bull; Not a certified industrial safety system.
    </footer>
</body>
</html>
"""
    out_file.write_text(html_content, encoding="utf-8")
    return out_file
