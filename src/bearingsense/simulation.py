# =============================================================================
# BearingSense — Physics-Informed Synthetic Bearing Vibration Simulator
# Engineered by : SISMAX (sismax01)
# =============================================================================
"""Physics-informed synthetic bearing vibration simulator.

Generates realistic, reproducible vibration acceleration signals modeled after the
Case Western Reserve University (CWRU) SKF 6205-2RS JEM Drive End bearing test rig.

IMPORTANT SCIENTIFIC INTEGRITY NOTICE:
Signals produced by this module are explicitly tagged with `source_type='synthetic_simulation'`.
They are designed for software verification, unit testing, CI/CD pipelines, and educational
demonstration of envelope spectrum demodulation without requiring a manual download of external
`.mat` files. Synthetic simulation results must never be reported as empirical real-world
benchmark metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
from scipy import io as spio

from bearingsense.bearing_kinematics import (
    BearingSpec,
    SKF_6205_DRIVE_END,
    compute_fault_frequencies,
)

FAULT_CLASSES: List[str] = [
    "Normal",
    "Outer_Race_Fault",
    "Inner_Race_Fault",
    "Ball_Fault",
]


def _synthesize_impulse_train_response(
    t: np.ndarray,
    fs: float,
    impact_freq_hz: float,
    modulation_freq_hz: float | None,
    modulation_depth: float,
    resonance_hz: float,
    damping_ratio: float,
    amplitude: float,
    rng: np.random.Generator,
    jitter_std_ratio: float = 0.022,
    subharmonic_alternation: bool = False,
    intermittent_drop_prob: float = 0.0,
) -> np.ndarray:
    """Synthesize damped structural resonance bursts triggered by bearing fault impacts."""
    duration = float(t[-1])
    mean_period = 1.0 / impact_freq_hz
    num_impacts = int(duration / mean_period) + 2

    # Generate impact arrival times with stochastic cage slip / jitter
    inter_arrival = rng.normal(
        loc=mean_period,
        scale=jitter_std_ratio * mean_period,
        size=num_impacts,
    )
    inter_arrival = np.clip(inter_arrival, 0.45 * mean_period, 1.55 * mean_period)
    impact_times = np.cumsum(inter_arrival)
    impact_times = impact_times[impact_times < duration]

    # Single-degree-of-freedom (SDOF) damped impulse response kernel (~12 ms duration)
    kernel_len = int(0.012 * fs)
    t_k = np.arange(kernel_len, dtype=np.float64) / fs
    omega_n = 2.0 * np.pi * resonance_hz
    omega_d = omega_n * np.sqrt(max(1.0 - damping_ratio ** 2, 0.01))
    kernel = np.exp(-damping_ratio * omega_n * t_k) * np.sin(omega_d * t_k)

    impulse_series = np.zeros_like(t, dtype=np.float64)
    impact_indices = (impact_times * fs).astype(int)
    valid_mask = (impact_indices >= 0) & (impact_indices < t.size)
    impact_indices = impact_indices[valid_mask]
    impact_times = impact_times[valid_mask]

    # Pulse-to-pulse stochastic amplitude variation
    pulse_amps = amplitude * rng.lognormal(mean=0.0, sigma=0.25, size=impact_indices.size)

    if intermittent_drop_prob > 0.0:
        # Rolling elements (balls) spin in 3D and intermittently rotate the spall out of contact
        active_mask = rng.uniform(0.0, 1.0, size=impact_indices.size) >= intermittent_drop_prob
        pulse_amps = pulse_amps * np.where(active_mask, 1.0, 0.12)

    if subharmonic_alternation:
        # For ball faults, alternating strikes on inner vs. outer race have different transfer paths
        alt_factors = np.where(np.arange(impact_indices.size) % 2 == 0, 1.0, 0.55)
        pulse_amps = pulse_amps * alt_factors

    if modulation_freq_hz is not None and modulation_freq_hz > 0:
        phase = rng.uniform(0, 2.0 * np.pi)
        mod_env = 1.0 + modulation_depth * np.cos(2.0 * np.pi * modulation_freq_hz * impact_times + phase)
        pulse_amps = pulse_amps * np.clip(mod_env, 0.08, 2.5)

    np.add.at(impulse_series, impact_indices, pulse_amps)
    response = np.convolve(impulse_series, kernel, mode="same")
    return response


def generate_synthetic_bearing_signal(
    fault_class: str,
    fs: float = 12000.0,
    duration_sec: float = 2.5,
    rpm: float = 1797.0,
    snr_db: float = 2.0,
    fault_severity: float = 1.0,
    resonance_hz: float = 3200.0,
    damping_ratio: float = 0.05,
    bearing: BearingSpec = SKF_6205_DRIVE_END,
    seed: int | None = 42,
) -> Dict:
    """Generate a single physics-informed synthetic bearing vibration signal.

    Args:
        fault_class: One of 'Normal', 'Outer_Race_Fault', 'Inner_Race_Fault', 'Ball_Fault'.
        fs: Sampling frequency in Hz (default 12,000 Hz matching CWRU DE baseline).
        duration_sec: Duration of recording in seconds.
        rpm: Shaft rotational speed in RPM.
        snr_db: Signal-to-noise ratio in dB between fault/baseline signal and additive noise.
        fault_severity: Relative fault impulse intensity multiplier.
        resonance_hz: Structural resonance center frequency in Hz.
        damping_ratio: Structural resonance damping ratio.
        bearing: BearingSpec geometry specification.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing the signal array and full physical metadata.
    """
    if fault_class not in FAULT_CLASSES:
        raise ValueError(f"Unknown fault_class '{fault_class}'. Expected one of {FAULT_CLASSES}.")
    if fs <= 0 or duration_sec <= 0 or rpm <= 0:
        raise ValueError("fs, duration_sec, and rpm must be strictly positive.")

    rng = np.random.default_rng(seed)
    n_samples = int(round(fs * duration_sec))
    t = np.arange(n_samples, dtype=np.float64) / fs

    # SISMAX Simulation Note: Include realistic induction motor slip variation (+/- 0.8%)
    effective_rpm = rpm * float(rng.uniform(0.992, 1.008))
    fault_freqs = compute_fault_frequencies(rpm=effective_rpm, bearing=bearing)
    fr = fault_freqs["fr"]

    # 1. Baseline shaft rotational harmonics + electromagnetic slot hum
    p1, p2, p3 = rng.uniform(0, 2.0 * np.pi, size=3)
    shaft_component = (
        0.095 * np.sin(2.0 * np.pi * fr * t + p1)
        + 0.048 * np.sin(2.0 * np.pi * (2.0 * fr) * t + p2)
        + 0.025 * np.sin(2.0 * np.pi * (3.0 * fr) * t + p3)
    )

    # Run-to-run variation in structural resonance frequency (+/- 160 Hz)
    run_res_hz = resonance_hz + rng.uniform(-160.0, 160.0)
    run_damping = np.clip(damping_ratio + rng.uniform(-0.01, 0.01), 0.025, 0.095)

    # 2. Fault-specific impact excitation + realistic cross-coupling background impulses
    if fault_class == "Normal":
        fault_component = _synthesize_impulse_train_response(
            t=t,
            fs=fs,
            impact_freq_hz=fr * float(rng.uniform(1.5, 4.2)),
            modulation_freq_hz=None,
            modulation_depth=0.0,
            resonance_hz=run_res_hz,
            damping_ratio=run_damping,
            amplitude=0.065 * fault_severity,
            rng=rng,
            jitter_std_ratio=0.38,
        )
    elif fault_class == "Outer_Race_Fault":
        fault_component = _synthesize_impulse_train_response(
            t=t,
            fs=fs,
            impact_freq_hz=fault_freqs["BPFO"],
            modulation_freq_hz=fr,
            modulation_depth=0.18,
            resonance_hz=run_res_hz,
            damping_ratio=run_damping,
            amplitude=0.72 * fault_severity,
            rng=rng,
            jitter_std_ratio=0.018,
            intermittent_drop_prob=0.06,
        )
    elif fault_class == "Inner_Race_Fault":
        fault_component = _synthesize_impulse_train_response(
            t=t,
            fs=fs,
            impact_freq_hz=fault_freqs["BPFI"],
            modulation_freq_hz=fr,
            modulation_depth=0.70,
            resonance_hz=run_res_hz,
            damping_ratio=run_damping,
            amplitude=0.66 * fault_severity,
            rng=rng,
            jitter_std_ratio=0.022,
            intermittent_drop_prob=0.10,
        )
    elif fault_class == "Ball_Fault":
        # Ball faults in real CWRU tests (Smith & Randall 2015) exhibit intermittent cage-slip
        # and occasional secondary raceway contact that makes them the hardest class to separate.
        primary_ball = _synthesize_impulse_train_response(
            t=t,
            fs=fs,
            impact_freq_hz=fault_freqs["BDF"],  # 2 * BSF
            modulation_freq_hz=fault_freqs["FTF"],
            modulation_depth=0.65,
            resonance_hz=run_res_hz,
            damping_ratio=run_damping,
            amplitude=0.58 * fault_severity,
            rng=rng,
            jitter_std_ratio=0.032,
            subharmonic_alternation=True,
            intermittent_drop_prob=0.16,
        )
        # Occasional secondary cage/raceway rattle
        secondary_rattle = _synthesize_impulse_train_response(
            t=t,
            fs=fs,
            impact_freq_hz=fault_freqs["BPFI"],
            modulation_freq_hz=fr,
            modulation_depth=0.40,
            resonance_hz=run_res_hz,
            damping_ratio=run_damping,
            amplitude=0.16 * fault_severity,
            rng=rng,
            jitter_std_ratio=0.08,
            intermittent_drop_prob=0.50,
        )
        fault_component = primary_ball + secondary_rattle
    else:
        fault_component = np.zeros_like(t)

    clean_signal = shaft_component + fault_component
    sig_power = float(np.mean(clean_signal ** 2))
    target_noise_power = max(sig_power / (10.0 ** (snr_db / 10.0)), 0.008)
    noise = rng.normal(0.0, np.sqrt(target_noise_power), size=n_samples)

    # Add random non-Gaussian mechanical background transients
    broadband_bursts = np.zeros(n_samples, dtype=np.float64)
    n_random_glitches = int(rng.integers(3, 12))
    glitch_locs = rng.integers(0, n_samples, size=n_random_glitches)
    broadband_bursts[glitch_locs] = rng.normal(0.0, 0.32, size=n_random_glitches)

    total_signal = clean_signal + noise + broadband_bursts

    return {
        "signal": total_signal,
        "time_sec": t,
        "fs": fs,
        "rpm": rpm,
        "label": fault_class,
        "fault_frequencies_hz": compute_fault_frequencies(rpm=rpm, bearing=bearing),
        "source_type": "synthetic_simulation",
    }


def generate_synthetic_benchmark_dataset(
    runs_per_class: int = 4,
    fs: float = 12000.0,
    duration_sec: float = 2.5,
    rpms: Sequence[float] = (1797.0, 1772.0, 1750.0, 1730.0),
    loads_hp: Sequence[int] = (0, 1, 2, 3),
    snr_db: float = 2.0,
    seed: int = 42,
) -> List[Dict]:
    """Generate a multi-run synthetic dataset across multiple operating loads and speeds.

    Each physical operating run receives a distinct `run_id` (e.g., `SYN_Outer_Race_Fault_run02_1772rpm`)
    and varied fault severity (incipient 0.007", moderate 0.014", severe 0.021") and SNR to mimic
    realistic experimental run-to-run variability.
    """
    rng = np.random.default_rng(seed)
    records: List[Dict] = []

    # Include an incipient low-amplitude run (0.52) so some windows near background noise challenge the classifiers
    severities = [0.95, 0.52, 1.15, 0.72]
    for cls_idx, fault_class in enumerate(FAULT_CLASSES):
        for run_idx in range(runs_per_class):
            rpm = float(rpms[run_idx % len(rpms)])
            load_hp = int(loads_hp[run_idx % len(loads_hp)])
            sev = severities[run_idx % len(severities)] * float(rng.uniform(0.88, 1.12))
            run_snr = snr_db + float(rng.uniform(-2.2, 2.0))
            run_seed = int(seed * 1000 + cls_idx * 100 + run_idx)

            sim = generate_synthetic_bearing_signal(
                fault_class=fault_class,
                fs=fs,
                duration_sec=duration_sec,
                rpm=rpm,
                snr_db=run_snr,
                fault_severity=sev,
                seed=run_seed,
            )
            run_id = f"SYN_{fault_class}_run{run_idx + 1:02d}_{int(rpm)}rpm"
            sim["run_id"] = run_id
            sim["load_hp"] = load_hp
            sim["fault_size_in"] = 0.0 if fault_class == "Normal" else (0.007 * ((run_idx % 3) + 1))
            records.append(sim)

    return records


def save_records_as_cwru_mat(records: Sequence[Dict], output_dir: str | Path) -> List[Path]:
    """Save vibration records in CWRU-compatible MATLAB `.mat` format for testing data loaders."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_files: List[Path] = []

    for idx, rec in enumerate(records, start=100):
        run_id = rec["run_id"]
        file_path = out_path / f"{run_id}.mat"
        mat_dict = {
            f"X{idx:03d}_DE_time": np.asarray(rec["signal"], dtype=np.float64).reshape(-1, 1),
            f"X{idx:03d}RPM": np.array([[float(rec["rpm"])]], dtype=np.float64),
            "label": str(rec["label"]),
            "fs": np.array([[float(rec["fs"])]], dtype=np.float64),
            "source_type": str(rec.get("source_type", "synthetic_simulation")),
        }
        spio.savemat(file_path, mat_dict)
        saved_files.append(file_path)

    return saved_files
