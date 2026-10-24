# =============================================================================
# BearingSense Test Suite — Evaluation, Explainability, CLI & Web App
# Engineered by : SISMAX (sismax01)
# =============================================================================
"""Unit and integration tests for leak-free evaluation, modeling, explainability, CLI, and web app."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from click.testing import CliRunner
import numpy as np

from bearingsense.cli import main
from bearingsense.evaluation import (
    create_stratified_group_folds,
    evaluate_models_cross_validation,
)
from bearingsense.explainability import (
    analyze_full_recording,
    compute_permutation_importance,
    explain_single_prediction,
)
from bearingsense.features import FEATURE_NAMES, extract_dataset_features
from bearingsense.modeling import (
    RandomForestBearingClassifier,
    load_model_bundle,
    save_model_bundle,
)
from bearingsense.simulation import (
    generate_synthetic_benchmark_dataset,
    save_records_as_cwru_mat,
)
from bearingsense.web_app import create_app


class TestEvaluationAndExplainability(unittest.TestCase):
    """Verify zero group leakage, model serialization, explainability, CLI commands, and web routes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.records = generate_synthetic_benchmark_dataset(
            runs_per_class=3, duration_sec=1.2, snr_db=4.0, seed=42
        )
        cls.df = extract_dataset_features(cls.records, window_size=2048, step_size=1024)

    def test_stratified_group_folds_have_zero_run_overlap(self) -> None:
        """Strictly verify that no `run_id` ever appears in both train and test within any fold."""
        folds = create_stratified_group_folds(self.df, n_splits=3, seed=42)
        self.assertEqual(len(folds), 3)

        for train_idx, test_idx in folds:
            train_runs = set(self.df.iloc[train_idx]["run_id"].unique())
            test_runs = set(self.df.iloc[test_idx]["run_id"].unique())
            self.assertEqual(len(train_runs & test_runs), 0, "Leakage detected between train and test runs!")
            self.assertEqual(len(self.df.iloc[test_idx]["label"].unique()), 4)

    def test_cross_validation_and_model_serialization(self) -> None:
        cv_res = evaluate_models_cross_validation(self.df, n_splits=3, seed=42, compare_leaky_split=True)
        self.assertIn("Random Forest", cv_res["models"])
        rf_f1 = cv_res["models"]["Random Forest"]["macro_f1"]
        self.assertGreater(rf_f1, 0.80)

        X = self.df[FEATURE_NAMES].to_numpy(dtype=np.float64)
        y = self.df["label"].to_numpy(dtype=str)
        rf = RandomForestBearingClassifier(n_estimators=25, max_depth=5, seed=42)
        rf.fit(X, y, feature_names=FEATURE_NAMES)

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "model.json"
            save_model_bundle(rf, X, y, bundle_path)
            loaded_rf, _ = load_model_bundle(bundle_path)

            orig_probs = rf.predict_proba(X[:5])
            load_probs = loaded_rf.predict_proba(X[:5])
            np.testing.assert_allclose(orig_probs, load_probs, atol=1e-7)

    def test_explainability_oof_global_local_and_recording_aggregation(self) -> None:
        X = self.df[FEATURE_NAMES].to_numpy(dtype=np.float64)
        y = self.df["label"].to_numpy(dtype=str)
        rf = RandomForestBearingClassifier(n_estimators=30, max_depth=5, seed=42)
        rf.fit(X, y, feature_names=FEATURE_NAMES)

        perm = compute_permutation_importance(
            rf, X, y, n_repeats=3, seed=42, df_for_oof=self.df, n_splits=3
        )
        self.assertEqual(len(perm), len(FEATURE_NAMES))
        self.assertGreater(perm[0]["importance_mean"], 0.0)
        self.assertEqual(perm[0]["evaluation_mode"], "out_of_fold_group_cv")

        norm_mask = y == "Normal"
        norm_mean = np.mean(X[norm_mask], axis=0)
        norm_std = np.maximum(np.std(X[norm_mask], axis=0), 1e-9)

        outer_idx = int(np.where(y == "Outer_Race_Fault")[0][0])
        expl = explain_single_prediction(rf, X[outer_idx], norm_mean, norm_std, rpm=1797.0)
        self.assertEqual(expl["predicted_class"], "Outer_Race_Fault")
        self.assertIn("BPFO", expl["physical_narrative"])
        self.assertEqual(len(expl["top_feature_attributions"]), 6)

        # Verify full-recording multi-window aggregation
        outer_rec = next(r for r in self.records if r["label"] == "Outer_Race_Fault")
        rec_expl, rep_win = analyze_full_recording(
            signal_1d=outer_rec["signal"],
            fs=float(outer_rec["fs"]),
            rpm=float(outer_rec["rpm"]),
            model=rf,
            normal_baseline_mean=norm_mean,
            normal_baseline_std=norm_std,
        )
        self.assertIn("recording_summary", rec_expl)
        self.assertGreaterEqual(rec_expl["recording_summary"]["total_windows_evaluated"], 10)
        self.assertEqual(rec_expl["recording_summary"]["overall_predicted_class"], "Outer_Race_Fault")
        self.assertEqual(rep_win.size, 2048)

    def test_cli_demo_evaluate_and_analyze_commands(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            demo_out = tmp_path / "demo_out"
            res = runner.invoke(
                main,
                ["demo", "--output-dir", str(demo_out), "--runs-per-class", "2", "--seed", "42"],
            )
            self.assertEqual(res.exit_code, 0, msg=res.output)
            self.assertTrue((demo_out / "diagnostic_report.html").exists())
            self.assertTrue((demo_out / "evaluation_metrics.json").exists())

            mat_dir = tmp_path / "mats"
            mat_files = save_records_as_cwru_mat(self.records, mat_dir)

            eval_out = tmp_path / "eval_out"
            res_ev = runner.invoke(
                main,
                ["evaluate", "--data-dir", str(mat_dir), "--output-dir", str(eval_out), "--folds", "3"],
            )
            self.assertEqual(res_ev.exit_code, 0, msg=res_ev.output)
            self.assertTrue((eval_out / "diagnostic_report.html").exists())

            analyze_out = tmp_path / "analyze_out"
            res_an = runner.invoke(
                main,
                [
                    "analyze",
                    "--input-file",
                    str(mat_files[0]),
                    "--model-path",
                    str(demo_out / "bearingsense_model_synthetic.json"),
                    "--output-dir",
                    str(analyze_out),
                ],
            )
            self.assertEqual(res_an.exit_code, 0, msg=res_an.output)

    def test_fastapi_web_app_routes(self) -> None:
        app = create_app()
        route_paths = {getattr(r, "path", "") for r in app.routes}
        self.assertIn("/", route_paths)
        self.assertIn("/api/health", route_paths)
        self.assertIn("/analyze-synthetic", route_paths)
        self.assertIn("/analyze-upload", route_paths)
        self.assertTrue(hasattr(app.state, "model"))
        self.assertEqual(len(app.state.model.classes_), 4)


if __name__ == "__main__":
    unittest.main()
