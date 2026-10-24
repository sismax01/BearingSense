"""Unit tests for bearing kinematics and digital signal processing modules."""

from __future__ import annotations

import unittest
import numpy as np

from bearingsense.bearing_kinematics import (
    BearingSpec,
    SKF_6203_FAN_END,
    SKF_6205_DRIVE_END,
    compute_fault_frequencies,
)
from bearingsense.signal_processing import (
    bandpass_filter,
    compute_envelope_spectrum,
    compute_harmonic_band_energy,
    compute_power_spectrum,
    validate_signal,
)


class TestBearingKinematics(unittest.TestCase):
    """Verify kinematic fault frequency formulas against published CWRU specifications."""

    def test_skf_6205_drive_end_multipliers(self) -> None:
        """CWRU Drive End SKF 6205-2RS JEM orders should match published constants."""
        mults = SKF_6205_DRIVE_END.fault_multipliers()
        self.assertAlmostEqual(mults["BPFO"], 3.5848, places=3)
        self.assertAlmostEqual(mults["BPFI"], 5.4152, places=3)
        self.assertAlmostEqual(mults["FTF"], 0.3983, places=3)
        self.assertAlmostEqual(mults["BDF"], 4.7135, places=3)

    def test_skf_6203_fan_end_multipliers(self) -> None:
        """CWRU Fan End SKF 6203-2RS JEM orders should match published constants."""
        mults = SKF_6203_FAN_END.fault_multipliers()
        self.assertAlmostEqual(mults["BPFO"], 3.0530, places=3)
        self.assertAlmostEqual(mults["BPFI"], 4.9469, places=3)
        self.assertAlmostEqual(mults["FTF"], 0.3816, places=3)
        self.assertAlmostEqual(mults["BDF"], 3.9874, places=3)

    def test_fault_frequencies_at_1797_rpm(self) -> None:
        """Verify exact Hz defect frequencies at 1797 RPM (f_r = 29.95 Hz)."""
        freqs = compute_fault_frequencies(rpm=1797.0, bearing=SKF_6205_DRIVE_END)
        self.assertAlmostEqual(freqs["fr"], 29.95, places=2)
        self.assertAlmostEqual(freqs["BPFO"], 107.36, places=1)
        self.assertAlmostEqual(freqs["BPFI"], 162.19, places=1)
        self.assertAlmostEqual(freqs["FTF"], 11.93, places=1)

    def test_invalid_rpm_or_geometry_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_fault_frequencies(rpm=0.0)
        with self.assertRaises(ValueError):
            compute_fault_frequencies(rpm=-1500.0)
        with self.assertRaises(ValueError):
            BearingSpec("Invalid", num_balls=2, ball_diameter_in=0.3, pitch_diameter_in=1.5)
        with self.assertRaises(ValueError):
            BearingSpec("Invalid", num_balls=8, ball_diameter_in=2.0, pitch_diameter_in=1.5)


class TestSignalProcessing(unittest.TestCase):
    """Verify bandpass filtering, power spectrum, and Hilbert envelope demodulation."""

    def test_validate_signal_errors(self) -> None:
        with self.assertRaises(ValueError):
            validate_signal(np.zeros((10, 10)), fs=12000.0)
        with self.assertRaises(ValueError):
            validate_signal(np.zeros(32), fs=12000.0)
        with self.assertRaises(ValueError):
            arr = np.ones(128)
            arr[5] = np.nan
            validate_signal(arr, fs=12000.0)
        with self.assertRaises(ValueError):
            validate_signal(np.ones(128), fs=-100.0)

    def test_bandpass_filter_isolates_inband_tone(self) -> None:
        fs = 12000.0
        t = np.arange(4096) / fs
        low_tone = 1.0 * np.sin(2.0 * np.pi * 120.0 * t)
        inband_tone = 1.0 * np.sin(2.0 * np.pi * 3200.0 * t)
        x = low_tone + inband_tone

        filtered = bandpass_filter(x, fs=fs, low_hz=2000.0, high_hz=5000.0)
        # RMS of single sine of amplitude 1.0 is ~0.7071; low_tone should be attenuated
        self.assertAlmostEqual(float(np.std(filtered[200:-200])), 1.0 / np.sqrt(2.0), delta=0.04)

        with self.assertRaises(ValueError):
            bandpass_filter(x, fs=fs, low_hz=4000.0, high_hz=2000.0)
        with self.assertRaises(ValueError):
            bandpass_filter(x, fs=fs, low_hz=1000.0, high_hz=7000.0)  # Above Nyquist

    def test_envelope_spectrum_detects_modulation_frequency(self) -> None:
        """An amplitude-modulated 3200 Hz carrier at 107.4 Hz should peak at ~107.4 Hz in the envelope spectrum."""
        fs = 12000.0
        t = np.arange(8192) / fs
        f_mod = 107.4
        f_carrier = 3200.0
        signal_am = (1.0 + 0.85 * np.cos(2.0 * np.pi * f_mod * t)) * np.sin(2.0 * np.pi * f_carrier * t)

        env_freqs, env_amps, _ = compute_envelope_spectrum(
            signal_am, fs=fs, bandpass_range=(2000.0, 5000.0), max_freq_hz=400.0
        )
        peak_freq = float(env_freqs[int(np.argmax(env_amps))])
        self.assertAlmostEqual(peak_freq, f_mod, delta=2.0)

        ratio_at_mod = compute_harmonic_band_energy(
            env_freqs, env_amps, fundamental_hz=f_mod, num_harmonics=2, tolerance_hz=4.0
        )
        ratio_off_band = compute_harmonic_band_energy(
            env_freqs, env_amps, fundamental_hz=162.2, num_harmonics=2, tolerance_hz=4.0
        )
        self.assertGreater(ratio_at_mod, 0.70)
        self.assertLess(ratio_off_band, 0.05)


if __name__ == "__main__":
    unittest.main()
