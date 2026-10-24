# =============================================================================
# BearingSense — Physics-Informed Feature Engineering (18 Features)
# Engineered by : SISMAX (sismax01)
# =============================================================================
"""Physics-informed feature extraction for bearing vibration condition monitoring.

Extracts 18 interpretable features per vibration window across three domains:
  1. Time-Domain Statistical & Impact Indicators (8 features)
  2. Frequency-Domain Spectral Shape & Resonance Ratios (4 features)
  3. Envelope Spectrum Characteristic Fault Frequency Indicators (6 features)
"""

from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np
import pandas as pd
from scipy import stats

from bearingsense.bearing_kinematics import (
    BearingSpec,
    SKF_6205_DRIVE_END,
    compute_fault_frequencies,
)
from bearingsense.signal_processing import (
    compute_envelope_spectrum,
    compute_harmonic_band_energy,
    compute_power_spectrum,
    validate_signal,
)

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    # Time-domain features
    "td_rms": "Root Mean Square (RMS) vibration amplitude — overall vibration energy level",
    "td_peak": "Peak absolute vibration amplitude — maximum instantaneous shock level",
    "td_kurtosis": "Excess fourth standardized moment (Kurtosis) — sensitivity to impulsive bearing impacts",
    "td_skewness": "Third standardized moment (Skewness) — asymmetry of the vibration waveform",
    "td_crest_factor": "Crest Factor (Peak / RMS) — ratio of peak impact to average vibration energy",
    "td_impulse_factor": "Impulse Factor (Peak / Mean Absolute Value) — indicator of sharp localized strikes",
    "td_clearance_factor": "Clearance Factor (Peak / Root-Mean-Square-Root^2) — sensitive to early spalling",
    "td_shape_factor": "Shape Factor (RMS / Mean Absolute Value) — waveform distribution shape",
    # Frequency-domain features
    "fd_spectral_centroid_hz": "Spectral Centroid (Hz) — center of gravity of the raw FFT power spectrum",
    "fd_spectral_spread_hz": "Spectral Spread (Hz) — bandwidth dispersion around the spectral centroid",
    "fd_spectral_entropy": "Normalized Shannon Spectral Entropy — flatness vs. tonal concentration of spectrum",
    "fd_high_freq_energy_ratio": "High-Frequency Resonance Ratio (2–5 kHz power / total power)",
    # Envelope-spectrum physical fault features
    "env_bpfo_energy_ratio": "Envelope Energy Ratio at Outer Race Fault Frequency (BPFO: 1x, 2x, 3x harmonics)",
    "env_bpfi_energy_ratio": "Envelope Energy Ratio at Inner Race Fault Frequency (BPFI: 1x, 2x, 3x harmonics)",
    "env_bsf_energy_ratio": "Envelope Energy Ratio at Ball Spin / Defect Frequency (BSF & 2xBSF harmonics)",
    "env_ftf_sideband_ratio": "Envelope Energy Ratio at Cage Train / Sideband Frequencies (FTF & BPFI±fr)",
    "env_kurtosis": "Kurtosis of the demodulated Hilbert envelope signal — cyclostationary impulsiveness",
    "env_dominant_freq_hz": "Dominant peak frequency (Hz) in the low-frequency demodulated envelope spectrum",
}

FEATURE_NAMES: List[str] = list(FEATURE_DESCRIPTIONS.keys())


def extract_features_from_window(
    window: np.ndarray,
    fs: float = 12000.0,
    rpm: float = 1797.0,
    bearing: BearingSpec = SKF_6205_DRIVE_END,
    bandpass_range: tuple[float, float] = (2000.0, 5000.0),
    tolerance_hz: float = 4.5,
) -> Dict[str, float]:
    """Extract all 18 domain-engineered features from a single 1D vibration window."""
    x = validate_signal(window, fs)
    x_zero = x - np.mean(x)
    abs_x = np.abs(x_zero)

    eps = 1e-12
    rms = float(np.sqrt(np.mean(x_zero ** 2))) + eps
    peak = float(np.max(abs_x))
    mean_abs = float(np.mean(abs_x)) + eps
    root_amp = float(np.mean(np.sqrt(abs_x)) ** 2) + eps

    kurt = float(stats.kurtosis(x_zero, fisher=True, bias=False))
    skew = float(stats.skew(x_zero, bias=False))
    crest_factor = peak / rms
    impulse_factor = peak / mean_abs
    clearance_factor = peak / root_amp
    shape_factor = rms / mean_abs

    # Frequency-domain features
    freqs, amps = compute_power_spectrum(x_zero, fs, apply_hann=True)
    power = (amps ** 2) + eps
    total_power = float(np.sum(power))

    prob = power / total_power
    centroid_hz = float(np.sum(freqs * prob))
    spread_hz = float(np.sqrt(np.sum(((freqs - centroid_hz) ** 2) * prob)))
    spectral_entropy = float(-np.sum(prob * np.log2(prob)) / np.log2(prob.size))

    nyq = 0.5 * fs
    hf_low = min(bandpass_range[0], 0.35 * nyq)
    hf_high = min(bandpass_range[1], 0.92 * nyq)
    hf_mask = (freqs >= hf_low) & (freqs <= hf_high)
    hf_energy_ratio = float(np.sum(power[hf_mask]) / total_power)

    # Envelope-spectrum fault frequency features (SISMAX kinematic mapping)
    env_freqs, env_amps, env_signal = compute_envelope_spectrum(
        x_zero, fs, bandpass_range=bandpass_range, max_freq_hz=800.0
    )
    fault_freqs = compute_fault_frequencies(rpm=rpm, bearing=bearing)

    bpfo_ratio = compute_harmonic_band_energy(
        env_freqs, env_amps, fault_freqs["BPFO"], num_harmonics=3, tolerance_hz=tolerance_hz
    )
    bpfi_ratio = compute_harmonic_band_energy(
        env_freqs, env_amps, fault_freqs["BPFI"], num_harmonics=3, tolerance_hz=tolerance_hz
    )
    bsf_ratio = compute_harmonic_band_energy(
        env_freqs, env_amps, fault_freqs["BSF"], num_harmonics=4, tolerance_hz=tolerance_hz
    )

    fr = fault_freqs["fr"]
    bpfi = fault_freqs["BPFI"]
    eval_mask = (env_freqs >= 5.0) & (env_freqs <= 800.0)
    if np.any(eval_mask):
        f_sub = env_freqs[eval_mask]
        p_sub = env_amps[eval_mask] ** 2
        tot_env_p = float(np.sum(p_sub)) + eps
        sb_centers = [
            fault_freqs["FTF"],
            2.0 * fault_freqs["FTF"],
            bpfi - fr,
            bpfi + fr,
            2.0 * bpfi - fr,
            2.0 * bpfi + fr,
        ]
        sb_mask = np.zeros_like(f_sub, dtype=bool)
        for c in sb_centers:
            if c > 5.0:
                sb_mask |= (f_sub >= c - tolerance_hz) & (f_sub <= c + tolerance_hz)
        ftf_sideband_ratio = float(np.sum(p_sub[sb_mask]) / tot_env_p)

        dom_mask = f_sub >= 10.0
        if np.any(dom_mask):
            f_dom = f_sub[dom_mask]
            p_dom = p_sub[dom_mask]
            dominant_freq_hz = float(f_dom[int(np.argmax(p_dom))])
        else:
            dominant_freq_hz = 0.0
    else:
        ftf_sideband_ratio = 0.0
        dominant_freq_hz = 0.0

    env_kurt = float(stats.kurtosis(env_signal - np.mean(env_signal), fisher=True, bias=False))

    return {
        "td_rms": rms,
        "td_peak": peak,
        "td_kurtosis": kurt,
        "td_skewness": skew,
        "td_crest_factor": crest_factor,
        "td_impulse_factor": impulse_factor,
        "td_clearance_factor": clearance_factor,
        "td_shape_factor": shape_factor,
        "fd_spectral_centroid_hz": centroid_hz,
        "fd_spectral_spread_hz": spread_hz,
        "fd_spectral_entropy": spectral_entropy,
        "fd_high_freq_energy_ratio": hf_energy_ratio,
        "env_bpfo_energy_ratio": bpfo_ratio,
        "env_bpfi_energy_ratio": bpfi_ratio,
        "env_bsf_energy_ratio": bsf_ratio,
        "env_ftf_sideband_ratio": ftf_sideband_ratio,
        "env_kurtosis": env_kurt,
        "env_dominant_freq_hz": dominant_freq_hz,
    }


def slice_signal_windows(
    signal_1d: np.ndarray,
    window_size: int = 2048,
    step_size: int = 1024,
) -> List[np.ndarray]:
    """Slice a continuous 1D vibration recording into fixed-length windows."""
    if window_size < 64:
        raise ValueError(f"window_size must be >= 64, got {window_size}")
    if step_size <= 0:
        raise ValueError(f"step_size must be > 0, got {step_size}")

    arr = np.asarray(signal_1d, dtype=np.float64)
    if arr.size < window_size:
        raise ValueError(
            f"Signal length ({arr.size}) is shorter than window_size ({window_size})."
        )

    windows: List[np.ndarray] = []
    for start in range(0, arr.size - window_size + 1, step_size):
        windows.append(arr[start : start + window_size])
    return windows


def extract_dataset_features(
    records: Sequence[Dict],
    window_size: int = 2048,
    step_size: int = 1024,
    bearing: BearingSpec = SKF_6205_DRIVE_END,
) -> pd.DataFrame:
    """Slice a collection of vibration recordings into windows and extract features.

    Preserves `run_id` metadata on every extracted window so that downstream evaluation
    can group by independent physical run (`StratifiedGroupKFold`) and prevent window-level
    data leakage.
    """
    rows: List[Dict] = []
    for rec in records:
        sig = rec["signal"]
        fs = float(rec.get("fs", 12000.0))
        rpm = float(rec.get("rpm", 1797.0))
        label = str(rec["label"])
        run_id = str(rec["run_id"])
        source_type = str(rec.get("source_type", "unknown"))
        load_hp = rec.get("load_hp", 0)
        fault_size_in = rec.get("fault_size_in", 0.0)

        windows = slice_signal_windows(sig, window_size=window_size, step_size=step_size)
        for win_idx, win in enumerate(windows):
            feat_dict = extract_features_from_window(
                win, fs=fs, rpm=rpm, bearing=bearing
            )
            row = {
                "run_id": run_id,
                "window_index": win_idx,
                "label": label,
                "rpm": rpm,
                "load_hp": load_hp,
                "fault_size_in": fault_size_in,
                "fs": fs,
                "source_type": source_type,
            }
            row.update(feat_dict)
            rows.append(row)

    if not rows:
        raise ValueError("No feature windows were extracted from the provided records.")
    return pd.DataFrame(rows)
