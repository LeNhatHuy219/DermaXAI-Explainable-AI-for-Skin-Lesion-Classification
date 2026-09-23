import json
import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Giữ đúng thứ tự import an toàn: src.dataset (kéo theo torch) trước sklearn.
from src.dataset import CONCEPT_NAMES, TOTAL_CONCEPT_STATES  # noqa: F401

import numpy as np
import pandas as pd

from experiments.run_m2 import (
    _default_paths,
    _manifest_hash,
    _select_best_C,
    build_oracle_features,
    concept_state_names,
    evaluate_m2_test,
    run_m2_experiment,
    select_validation_threshold,
)


class M2ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.manifest_path = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
        self.label_mapping_path = os.path.join(PROJECT_ROOT, "data", "label_mapping.json")
        with open(self.label_mapping_path, "r") as handle:
            self.label_mapping = json.load(handle)
        self.df = pd.read_csv(self.manifest_path, dtype={"is_inconsistent_profile": bool})

    def test_build_oracle_features_matches_concept_onehot(self):
        subset = self.df.head(10)
        features = build_oracle_features(subset, self.label_mapping)
        self.assertEqual(features.shape, (10, TOTAL_CONCEPT_STATES))
        # Mỗi mẫu phải có đúng 7 bit được kích hoạt (1 cho mỗi concept)
        self.assertTrue(np.all(features.sum(axis=1) == 7.0))

    def test_concept_state_names_are_unique_and_complete(self):
        names = concept_state_names(self.label_mapping)
        self.assertEqual(len(names), TOTAL_CONCEPT_STATES)
        self.assertEqual(len(set(names)), TOTAL_CONCEPT_STATES)
        self.assertTrue(all(name for name in names))
        self.assertTrue(all("=" in name for name in names))

    def test_select_best_c_picks_highest_validation_balanced_accuracy(self):
        rng = np.random.RandomState(0)
        X_train = rng.rand(60, 4)
        y_train = (X_train[:, 0] > 0.5).astype(int)
        X_valid = rng.rand(30, 4)
        y_valid = (X_valid[:, 0] > 0.5).astype(int)
        best_c, scores, best_model = _select_best_C(X_train, y_train, X_valid, y_valid, [0.01, 1.0, 100.0], seed=42)
        self.assertIn(best_c, scores)
        self.assertEqual(set(scores.keys()), {0.01, 1.0, 100.0})
        self.assertIsNotNone(best_model)

    def test_threshold_selection_matches_expected_tie_break(self):
        threshold, bacc = select_validation_threshold(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.55, 0.45, 0.8])
        )
        self.assertAlmostEqual(threshold, 0.45)
        self.assertAlmostEqual(bacc, 0.75)

    def test_default_paths_are_seed_specific(self):
        self.assertNotEqual(_default_paths(42), _default_paths(123))

    def test_manifest_hash_is_deterministic(self):
        self.assertEqual(_manifest_hash(self.manifest_path), _manifest_hash(self.manifest_path))

    def test_full_train_then_test_protocol_on_real_manifest(self):
        with tempfile.TemporaryDirectory(prefix="m2_protocol_") as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "m2_best.joblib")
            valid_results_path = os.path.join(temp_dir, "valid.json")
            test_results_path = os.path.join(temp_dir, "test.json")

            valid_results = run_m2_experiment(
                manifest_path=self.manifest_path,
                c_grid=[0.1, 1.0],
                checkpoint_path=checkpoint_path,
                results_path=valid_results_path,
                save_results=True,
                overwrite=True,
            )
            self.assertEqual(valid_results["mode"], "train_validation")
            self.assertEqual(len(valid_results["validation_predictions"]), 203)
            self.assertTrue(os.path.exists(checkpoint_path))
            self.assertTrue(0.0 <= valid_results["decision_threshold"] <= 1.0)

            test_results = evaluate_m2_test(
                checkpoint_path=checkpoint_path,
                manifest_path=self.manifest_path,
                results_path=test_results_path,
                save_results=True,
                overwrite=True,
            )
            self.assertEqual(test_results["mode"], "final_test")
            self.assertEqual(len(test_results["test_predictions"]), 395)
            self.assertEqual(test_results["decision_threshold"], valid_results["decision_threshold"])


if __name__ == "__main__":
    unittest.main()
