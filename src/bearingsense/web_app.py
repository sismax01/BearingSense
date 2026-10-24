# =============================================================================
# BearingSense — Interactive FastAPI Web Workbench
# Engineered by : SISMAX (sismax01)
# =============================================================================
"""Interactive FastAPI web dashboard for BearingSense.

Allows users to launch a local browser interface (`bearingsense serve --port 8000`) to:
  1. Generate and inspect synthetic bearing vibration signals under custom RPM, SNR, and fault types.
  2. Upload real CWRU `.mat` or `.csv` vibration files for full-recording multi-window
     classification, envelope spectrum analysis, and local feature attribution.
"""

from __future__ import annotations

import base64
from pathlib import Path
import tempfile
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import matplotlib
matplotlib.use("Agg")
import numpy as np

from bearingsense.bearing_kinematics import SKF_6205_DRIVE_END
from bearingsense.data_loader import load_csv_vibration_file, load_cwru_mat_file
from bearingsense.explainability import analyze_full_recording
from bearingsense.features import FEATURE_NAMES, extract_dataset_features
from bearingsense.modeling import RandomForestBearingClassifier
from bearingsense.simulation import (
    FAULT_CLASSES,
    generate_synthetic_bearing_signal,
    generate_synthetic_benchmark_dataset,
)
from bearingsense.visualization import plot_single_window_inspection


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance with a pre-warmed reference model."""
    app = FastAPI(
        title="BearingSense Web Explorer — by SISMAX",
        description="Explainable Machine Learning & Envelope Spectrum Diagnostics for Industrial Bearings",
        version="0.1.0",
    )

    records = generate_synthetic_benchmark_dataset(runs_per_class=4, duration_sec=2.0, seed=42)
    df_train = extract_dataset_features(records, window_size=2048, step_size=1024)
    X_tr = df_train[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_tr = df_train["label"].to_numpy(dtype=str)

    model = RandomForestBearingClassifier(n_estimators=50, max_depth=6, seed=42)
    model.fit(X_tr, y_tr, feature_names=FEATURE_NAMES)

    normal_mask = y_tr == "Normal"
    norm_mean = np.mean(X_tr[normal_mask], axis=0)
    norm_std = np.maximum(np.std(X_tr[normal_mask], axis=0), 1e-9)

    app.state.model = model
    app.state.norm_mean = norm_mean
    app.state.norm_std = norm_std

    @app.get("/api/health")
    def health_check() -> JSONResponse:
        return JSONResponse({
            "status": "ok",
            "application": "BearingSense",
            "author_signature": "SISMAX (sismax01)",
            "version": "0.1.0",
            "classes": FAULT_CLASSES,
            "bearing": SKF_6205_DRIVE_END.designation,
        })

    @app.get("/", response_class=HTMLResponse)
    def index_page() -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>BearingSense — Interactive Vibration Diagnostics (SISMAX)</title>
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; background: #f7fafc; color: #1a202c; margin: 0; }
        header { background: #1e3a5f; color: white; padding: 1.5rem 2rem; }
        .container { max-width: 1150px; margin: 1.5rem auto; padding: 0 1.5rem; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
        .card { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.5rem; }
        label { display: block; font-weight: 600; margin-top: 0.75rem; font-size: 0.9rem; }
        select, input { width: 100%; padding: 0.5rem; margin-top: 0.25rem; border: 1px solid #cbd5e0; border-radius: 4px; }
        button { background: #2b6cb0; color: white; border: none; padding: 0.65rem 1.2rem; border-radius: 4px; font-weight: 600; cursor: pointer; margin-top: 1rem; }
        button:hover { background: #1e4e8c; }
        footer { text-align: center; padding: 1.2rem; color: #4a5568; font-size: 0.85rem; }
    </style>
</head>
<body>
    <header>
        <h1 style="margin:0;">BearingSense — Interactive Diagnostic Workbench</h1>
        <p style="margin:0.3rem 0 0 0; opacity:0.9;">SKF 6205-2RS JEM Drive End Bearing &bull; Hilbert Envelope Demodulation &amp; Explainable ML &bull; Engineered by SISMAX</p>
    </header>
    <div class="container">
        <div class="grid">
            <div class="card">
                <h2 style="margin-top:0;">Option A: Simulate Synthetic Bearing Condition</h2>
                <p style="font-size:0.9rem; color:#4a5568;">Synthesize a physics-informed vibration signal to inspect envelope fault harmonics and local ML attributions.</p>
                <form action="/analyze-synthetic" method="post">
                    <label>Simulated Bearing Condition</label>
                    <select name="fault_class">
                        <option value="Outer_Race_Fault">Outer Race Fault (BPFO Impact Train)</option>
                        <option value="Inner_Race_Fault">Inner Race Fault (BPFI + Shaft Modulation)</option>
                        <option value="Ball_Fault">Ball Fault (2xBSF + Cage Modulation)</option>
                        <option value="Normal">Normal Healthy Baseline</option>
                    </select>
                    <label>Shaft Speed (RPM)</label>
                    <input type="number" name="rpm" value="1797" min="600" max="3600" step="1">
                    <label>Signal-to-Noise Ratio (SNR dB)</label>
                    <input type="number" name="snr_db" value="4.0" min="-5" max="25" step="0.5">
                    <button type="submit">Synthesize &amp; Diagnose Signal</button>
                </form>
            </div>
            <div class="card">
                <h2 style="margin-top:0;">Option B: Upload CWRU .mat or .csv Recording</h2>
                <p style="font-size:0.9rem; color:#4a5568;">Upload an experimental MATLAB (<code>.mat</code>) file from the CWRU Bearing Data Center or a 1D vibration <code>.csv</code> file.</p>
                <form action="/analyze-upload" method="post" enctype="multipart/form-data">
                    <label>Vibration Data File (.mat or .csv)</label>
                    <input type="file" name="file" accept=".mat,.csv" required>
                    <label>Sampling Rate (Hz)</label>
                    <input type="number" name="fs" value="12000" min="1000" max="96000" step="100">
                    <label>Default Shaft Speed (RPM, if not in .mat)</label>
                    <input type="number" name="rpm" value="1797" min="600" max="3600" step="1">
                    <button type="submit">Analyze Uploaded Recording</button>
                </form>
            </div>
        </div>
    </div>
    <footer>BearingSense v0.1.0 &bull; Engineered by <strong>SISMAX (sismax01)</strong></footer>
</body>
</html>"""

    @app.post("/analyze-synthetic", response_class=HTMLResponse)
    def analyze_synthetic(
        fault_class: str = Form("Outer_Race_Fault"),
        rpm: float = Form(1797.0),
        snr_db: float = Form(4.0),
    ) -> str:
        sim = generate_synthetic_bearing_signal(
            fault_class=fault_class, fs=12000.0, duration_sec=1.2, rpm=rpm, snr_db=snr_db, seed=101
        )
        return _render_analysis_result(
            app=app,
            signal_1d=sim["signal"],
            fs=12000.0,
            rpm=rpm,
            source_label=f"Synthetic Simulation ({fault_class}, {int(rpm)} RPM, SNR={snr_db} dB)",
            source_type="synthetic_simulation",
        )

    @app.post("/analyze-upload", response_class=HTMLResponse)
    async def analyze_upload(
        file: UploadFile = File(...),
        fs: float = Form(12000.0),
        rpm: float = Form(1797.0),
    ) -> HTMLResponse:
        suffix = Path(file.filename or "upload.mat").suffix.lower()
        content = await file.read()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / f"uploaded{suffix}"
            tmp_path.write_bytes(content)
            try:
                if suffix == ".mat":
                    rec = load_cwru_mat_file(tmp_path, default_fs=fs, default_rpm=rpm)
                elif suffix == ".csv":
                    rec = load_csv_vibration_file(tmp_path, fs=fs, rpm=rpm)
                else:
                    return HTMLResponse("<h3>Unsupported file extension. Please upload a .mat or .csv file.</h3>", status_code=400)
            except Exception as exc:
                return HTMLResponse(f"<h3>Error reading uploaded file: {exc}</h3>", status_code=400)

        html_out = _render_analysis_result(
            app=app,
            signal_1d=rec["signal"],
            fs=float(rec["fs"]),
            rpm=float(rec["rpm"]),
            source_label=f"Uploaded File: {file.filename}",
            source_type=str(rec.get("source_type", "uploaded_file")),
        )
        return HTMLResponse(html_out)

    return app


def _render_analysis_result(
    app: FastAPI,
    signal_1d: np.ndarray,
    fs: float,
    rpm: float,
    source_label: str,
    source_type: str,
) -> str:
    explanation, rep_win = analyze_full_recording(
        signal_1d=signal_1d,
        fs=fs,
        rpm=rpm,
        model=app.state.model,
        normal_baseline_mean=app.state.norm_mean,
        normal_baseline_std=app.state.norm_std,
    )
    rec_sum = explanation["recording_summary"]
    with tempfile.TemporaryDirectory() as tmpdir:
        png_path = Path(tmpdir) / "inspect.png"
        plot_single_window_inspection(
            window=rep_win,
            fs=fs,
            rpm=rpm,
            explanation=explanation,
            output_path=png_path,
            source_type=source_type,
        )
        b64_img = base64.b64encode(png_path.read_bytes()).decode("ascii")

    rows = ""
    for a in explanation["top_feature_attributions"]:
        rows += f"<tr><td><code>{a['feature']}</code></td><td>{a['description']}</td><td>{a['observed_value']:.4f}</td><td>{a['z_score_vs_healthy']:+.2f}&sigma;</td><td><strong>{a['probability_contribution']:+.3f}</strong></td></tr>"

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>BearingSense Diagnostic Result — SISMAX</title>
    <style>
        body {{ font-family: system-ui, sans-serif; background: #f7fafc; color: #1a202c; margin: 0; padding: 1.5rem; }}
        .container {{ max-width: 1150px; margin: 0 auto; background: white; padding: 1.5rem; border-radius: 8px; border: 1px solid #e2e8f0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }}
        th, td {{ padding: 0.6rem; border-bottom: 1px solid #e2e8f0; text-align: left; }}
        th {{ background: #edf2f7; }}
        .narrative {{ background: #ebf8ff; border-left: 4px solid #2b6cb0; padding: 1rem; margin: 1rem 0; }}
    </style>
</head>
<body>
    <div class="container">
        <p><a href="/">&larr; Back to BearingSense Workbench</a></p>
        <h2>Diagnostic Result — {source_label}</h2>
        <p><strong>Recording Summary:</strong> Evaluated <strong>{rec_sum['total_windows_evaluated']}</strong> windows &bull;
           Overall Class: <strong>{rec_sum['overall_predicted_class']}</strong> ({rec_sum['mean_class_confidence']:.1%} mean confidence, {rec_sum['window_agreement_ratio']:.1%} window agreement)</p>
        <div class="narrative"><strong>Physical Explanation (Representative Window #{rec_sum['representative_window_index']}):</strong> {explanation['physical_narrative']}</div>
        <div style="text-align:center;"><img src="data:image/png;base64,{b64_img}" style="max-width:100%; border:1px solid #cbd5e0; border-radius:6px;"></div>
        <h3>Top Feature Attributions</h3>
        <table>
            <thead><tr><th>Feature</th><th>Description</th><th>Observed</th><th>Z-Score vs. Healthy</th><th>Contribution (&Delta;P)</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
</body>
</html>"""
