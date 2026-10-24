"""Unit tests for feature extraction, synthetic simulation, and CWRU .mat/.csv data loaders."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from bearingsense.data_loader import (
    infer_label_from_filename,
    load_csv_vibration_file,
    load_cwru_mat_file,
    load_directory_dataset,
)
from bearingsense.features import (
    FEATURE_NAMES,
    extract_dataset_features,
    extract_features_from_window,
    slice_signal_windows,
)
from bearingsense.simulation import (
    FAULT_CLASSES,
    generate_synthetic_bearing_signal,
    generate_synthetic_benchmark_dataset,
    save_records_as_cwru_mat,
)


class TestFeaturesAndLoaders(unittest.TestCase):
    """Test feature extraction physical discrimination and CWRU file ingestion."""

    def test_feature_extraction_returns_all_18_finite_features(self) -> None:
        sim = generate_synthetic_bearing_signal("Outer_Race_Fault", duration_sec=0.5, seed=10)
        win = sim["signal"][:2048]
        feats = extract_features_from_window(win, fs=sim["fs"], rpm=sim["rpm"])

        self.assertEqual(set(feats.keys()), set(FEATURE_NAMES))
        self.assertEqual(len(feats), 18)
        for k, v in feats.items():
            self.assertTrue(np.isfinite(v), f"Feature {k} is not finite: {v}")

    def test_physical_fault_features_match_simulated_condition(self) -> None:
        """Outer race fault should have highest BPFO ratio; inner race fault highest BPFI ratio."""
        sim_outer = generate_synthetic_bearing_signal("Outer_Race_Fault", duration_sec=0.5, snr_db=8.0, seed=11)
        sim_inner = generate_synthetic_bearing_signal("Inner_Race_Fault", duration_sec=0.5, snr_db=8.0, seed=12)
        sim_norm = generate_synthetic_bearing_signal("Normal", duration_sec=0.5, snr_db=8.0, seed=13)

        f_outer = extract_features_from_window(sim_outer["signal"][:2048], fs=12000.0, rpm=1797.0)
        f_inner = extract_features_from_window(sim_inner["signal"][:2048], fs=12000.0, rpm=1797.0)
        f_norm = extract_features_from_window(sim_norm["signal"][:2048], fs=12000.0, rpm=1797.0)

        self.assertGreater(f_outer["env_bpfo_energy_ratio"], f_inner["env_bpfo_energy_ratio"])
        self.assertGreater(f_outer["env_bpfo_energy_ratio"], f_norm["env_bpfo_energy_ratio"])
        self.assertGreater(f_inner["env_bpfi_energy_ratio"], f_outer["env_bpfi_energy_ratio"])
        self.assertGreater(f_outer["td_rms"], f_norm["td_rms"])

    def test_slice_signal_windows_and_validation(self) -> None:
        sig = np.ones(5000)
        wins = slice_signal_windows(sig, window_size=2048, step_size=1024)
        self.assertEqual(len(wins), 3)
        self.assertEqual(wins[0].size, 2048)

        with self.assertRaises(ValueError):
            slice_signal_windows(np.ones(500), window_size=2048, step_size=1024)

    def test_infer_label_from_cwru_catalog_and_names(self) -> None:
        self.assertEqual(infer_label_from_filename("97.mat"), "Normal")
        self.assertEqual(infer_label_from_filename("105.mat"), "Inner_Race_Fault")
        self.assertEqual(infer_label_from_filename("118.mat"), "Ball_Fault")
        self.assertEqual(infer_label_from_filename("130.mat"), "Outer_Race_Fault")
        self.assertEqual(infer_label_from_filename("SYN_Outer_Race_Fault_run01.mat"), "Outer_Race_Fault")

    def test_save_and_load_cwru_mat_and_csv_files(self) -> None:
        records = generate_synthetic_benchmark_dataset(runs_per_class=2, duration_sec=0.5, seed=99)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            mat_files = save_records_as_cwru_mat(records, tmp_path)
            self.assertEqual(len(mat_files), 8)

            loaded_single = load_cwru_mat_file(mat_files[0])
            self.assertIn("signal", loaded_single)
            self.assertEqual(loaded_single["label"], records[0]["label"])

            loaded_all = load_directory_dataset(tmp_path)
            self.assertEqual(len(loaded_all), 8)

            df_feats = extract_dataset_features(loaded_all, window_size=2048, step_size=1024)
            self.assertEqual(set(df_feats["label"].unique()), set(FAULT_CLASSES))

            # Also verify CSV loading
            csv_path = tmp_path / "Normal_test.csv"
            pd.DataFrame({"time_s": np.arange(2500) / 12000.0, "acceleration": records[0]["signal"][:2500]}).to_csv(
                csv_path, index=False
            )
            csv_rec = load_csv_vibration_file(csv_path, fs=12000.0, rpm=1797.0)
            self.assertEqual(csv_rec["label"], "Normal")
            self.assertEqual(csv_rec["signal"].size, 2500)


if __name__ == "__main__":
    unittest.main()
