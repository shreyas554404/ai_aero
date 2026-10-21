import tempfile
import unittest
from pathlib import Path

from predictive_maintenance import PredictiveMaintenanceModel


class PredictiveMaintenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pm = PredictiveMaintenanceModel()

    def test_synthetic_data_has_required_columns(self) -> None:
        df = self.pm.generate_synthetic_data(n_samples=50, random_seed=7)
        required = {"flight_id", "manufacturer", "component", "vibration", "temperature", "pressure", "cycles", "wear_index"}
        self.assertTrue(required.issubset(set(df.columns)))

    def test_training_produces_best_model_for_each_component(self) -> None:
        df = self.pm.generate_synthetic_data(n_samples=220, random_seed=2)
        metrics = self.pm.train_models(df)
        self.assertGreater(len(metrics), 0)
        for comp in sorted(df["component"].unique()):
            self.assertIn(comp, self.pm.best_model_per_component)

    def test_limit_breach_marks_component_for_maintenance(self) -> None:
        df = self.pm.generate_synthetic_data(n_samples=260, random_seed=3)
        self.pm.train_models(df)
        sample = df.iloc[[0]].copy()
        m = sample.iloc[0]["manufacturer"]
        c = sample.iloc[0]["component"]
        sample.loc[sample.index[0], "temperature"] = self.pm.limits[m][c]["temperature_max"] + 100

        scored, summary = self.pm.evaluate_components(sample)
        self.assertFalse(bool(scored.iloc[0]["airworthy"]))
        self.assertEqual(summary.iloc[0]["final_decision"], "SEND_TO_MAINTENANCE")

    def test_graph_generation_outputs_files(self) -> None:
        df = self.pm.generate_synthetic_data(n_samples=180, random_seed=5)
        self.pm.train_models(df)
        scored, _ = self.pm.evaluate_components(df.head(80))
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self.pm.generate_3d_graphs(scored, output_dir=tmpdir)
            self.assertGreaterEqual(len(files), 2)
            for file in files:
                self.assertTrue(Path(file).exists())


if __name__ == "__main__":
    unittest.main()
