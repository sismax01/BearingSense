# =============================================================================
# BearingSense — Digital Signal Processing & Hilbert Envelope Demodulation
# Engineered by : SISMAX (sismax01)
# =============================================================================
"""Digital signal processing routines for bearing vibration analysis.

Includes:
  - Zero-phase Butterworth bandpass filtering (resonant band isolation)
  - Analytic signal demodulation via the Hilbert transform
  - Envelope spectrum (rectified analytic envelope FFT) computation
  - One-sided amplitude spectrum
  - Harmonic band energy extraction around characteristic fault frequencies
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy import signal


def validate_signal(x: np.ndarray, fs: float) -> np.ndarray:
    """Validate 1D vibration signal array and sampling frequency."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D vibration signal array, got {arr.ndim}D array.")
    if arr.size < 64:
        raise ValueError(f"Signal length must be at least 64 samples, got {arr.size}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Signal contains NaN or infinite values.")
    if fs <= 0:
        raise ValueError(f"Sampling frequency fs must be positive, got {fs}.")
    return arr


def bandpass_filter(
    x: np.ndarray,
    fs: float,
    low_hz: float = 2000.0,
    high_hz: float = 5000.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to isolate structural resonance.

    Args:
        x: 1D vibration acceleration signal.
        fs: Sampling frequency in Hz.
        low_hz: Lower cutoff frequency in Hz.
        high_hz: Upper cutoff frequency in Hz.
        order: Butterworth filter order.

    Returns:
        Bandpass-filtered 1D numpy array of same length as x.
    """
    arr = validate_signal(x, fs)
    nyquist = 0.5 * fs
    if low_hz <= 0 or high_hz >= nyquist or low_hz >= high_hz:
        raise ValueError(
            f"Invalid bandpass range [{low_hz}, {high_hz}] Hz for Nyquist frequency {nyquist} Hz."
        )
    # SISMAX DSP Note: Second-order sections (sosfiltfilt) ensure numerical stability
    # and zero phase distortion across the bearing housing resonance band.
    sos = signal.butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, arr)


def compute_power_spectrum(
    x: np.ndarray,
    fs: float,
    apply_hann: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute one-sided amplitude spectrum of a vibration signal.

    Args:
        x: 1D vibration signal.
        fs: Sampling frequency in Hz.
        apply_hann: Whether to apply a Hann window before FFT to reduce spectral leakage.

    Returns:
        Tuple of (frequencies_hz, amplitude_spectrum).
    """
    arr = validate_signal(x, fs)
    arr_zero_mean = arr - np.mean(arr)
    n = arr_zero_mean.size

    if apply_hann:
        win = np.hanning(n)
        scale = 2.0 / np.sum(win)
        windowed = arr_zero_mean * win
    else:
        scale = 2.0 / n
        windowed = arr_zero_mean

    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    amps = np.abs(fft_vals) * scale
    if amps.size > 0:
        amps[0] *= 0.5  # DC component is not doubled
    return freqs, amps


def compute_envelope_signal(
    x: np.ndarray,
    fs: float,
    bandpass_range: Tuple[float, float] | None = (2000.0, 5000.0),
    filter_order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the bandpass-filtered signal and its analytic Hilbert envelope."""
    arr = validate_signal(x, fs)
    if bandpass_range is not None:
        low_hz, high_hz = bandpass_range
        nyq = 0.5 * fs
        effective_high = min(high_hz, 0.92 * nyq)
        effective_low = min(low_hz, 0.45 * effective_high)
        filtered = bandpass_filter(arr, fs, low_hz=effective_low, high_hz=effective_high, order=filter_order)
    else:
        filtered = arr - np.mean(arr)

    analytic = signal.hilbert(filtered)
    envelope = np.abs(analytic)
    return filtered, envelope


def compute_envelope_spectrum(
    x: np.ndarray,
    fs: float,
    bandpass_range: Tuple[float, float] | None = (2000.0, 5000.0),
    max_freq_hz: float | None = 1000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Hilbert envelope spectrum for bearing defect detection.

    Demodulates high-frequency resonance excited by repetitive bearing fault impacts:
      1. Bandpass filters the raw signal in the structural resonance band.
      2. Extracts the instantaneous amplitude envelope |hilbert(x_filtered)|.
      3. Removes the DC offset of the envelope and computes its one-sided amplitude spectrum.
    """
    _, envelope = compute_envelope_signal(x, fs, bandpass_range=bandpass_range)
    env_ac = envelope - np.mean(envelope)
    freqs, amps = compute_power_spectrum(env_ac, fs, apply_hann=True)

    if max_freq_hz is not None and max_freq_hz > 0:
        mask = freqs <= max_freq_hz
        freqs = freqs[mask]
        amps = amps[mask]

    return freqs, amps, envelope


def compute_harmonic_band_energy(
    env_freqs: np.ndarray,
    env_amps: np.ndarray,
    fundamental_hz: float,
    num_harmonics: int = 3,
    tolerance_hz: float = 4.5,
    max_eval_freq_hz: float = 800.0,
) -> float:
    """Compute normalized envelope spectral energy concentrated around a fault frequency and its harmonics.

    Accounts for slight shaft speed slip (typically 1-2% in induction motors) by integrating
    squared envelope amplitudes within [k * f_fault - tolerance_hz, k * f_fault + tolerance_hz]
    for harmonics k = 1 .. num_harmonics, normalized by the total envelope spectrum energy
    up to max_eval_freq_hz.
    """
    if fundamental_hz <= 0 or env_freqs.size == 0:
        return 0.0

    eval_mask = (env_freqs >= 5.0) & (env_freqs <= max_eval_freq_hz)
    if not np.any(eval_mask):
        return 0.0

    f_sub = env_freqs[eval_mask]
    power_sub = env_amps[eval_mask] ** 2
    total_energy = float(np.sum(power_sub))
    if total_energy <= 1e-15:
        return 0.0

    band_mask = np.zeros_like(f_sub, dtype=bool)
    for k in range(1, num_harmonics + 1):
        center = k * fundamental_hz
        band_mask |= (f_sub >= (center - tolerance_hz)) & (f_sub <= (center + tolerance_hz))

    harmonic_energy = float(np.sum(power_sub[band_mask]))
    return harmonic_energy / total_energy
