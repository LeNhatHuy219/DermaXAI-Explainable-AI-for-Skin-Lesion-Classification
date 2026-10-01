"""Analytical checks for exhaustive intervention aggregation and run compatibility."""
import copy
import unittest
from pathlib import Path

import torch
import numpy as np
from src.dataset import CONCEPT_NAMES, CONCEPT_NUM_CLASSES
from src.intervention import evaluate_soft_interventions
from src.protocol import load_concept_schema, manifest_fingerprint
from experiments.run_m3_interventions import summarize_interventions

ROOT = Path(__file__).resolve().parents[1]


class InterventionSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = ROOT / "data/manifest.csv"
        schema = load_concept_schema(str(manifest))
        probabilities = np.concatenate([np.full((4, k), 1/k)
                                        for k in CONCEPT_NUM_CLASSES.values()], axis=1)
        labels = [0, 1, 0, 1]
        targets = np.zeros((4, 7), dtype=int)
        targets[:, 0] = labels
        # Only the first concept matters. Its correction gives perfect diagnosis;
        # all other corrections leave a constant baseline with BA=0.5.
        head = torch.nn.Linear(28, 2)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
            head.weight[1, 0], head.weight[1, 1] = -2., 2.
        cls.runs = []
        for seed, threshold in zip([42, 123, 2026], [.4, .5, .6]):
            run = evaluate_soft_interventions(head, probabilities, targets, labels, schema,
                                             threshold, [10, 20, 30, 40])
            run.update(model="M3_SoftJointCBM", mode="intervention_analysis",
                       hyperparameters={"seed": seed, "protocol": "fixture"},
                       concept_schema=schema, manifest_fingerprint=manifest_fingerprint(str(manifest)),
                       checkpoint_sha256=str(seed))
            cls.runs.append(run)

    def test_curve_matches_analytic_subset_probability_and_auc(self):
        result = summarize_interventions(self.runs)
        for row in result["curve"]:
            # Fraction of size-m subsets containing the first group is m/7.
            self.assertAlmostEqual(row["balanced_accuracy"]["mean"], .5+.5*row["m"]/7)
            self.assertAlmostEqual(row["balanced_accuracy"]["sample_sd"], 0.)
            self.assertAlmostEqual(row["improved_cases"]["mean"], 2*row["m"]/7)
            self.assertEqual(row["worsened_cases"]["mean"], 0.)
        auc = result["intervention_curve_auc"]["balanced_accuracy"]
        self.assertAlmostEqual(auc["absolute"]["mean"], .75)
        self.assertAlmostEqual(auc["gain_over_baseline"]["mean"], .25)
        self.assertAlmostEqual(result["per_concept"][CONCEPT_NAMES[0]]["delta_balanced_accuracy"]["mean"], .5)
        self.assertIsNone(result["curve"][-1]["remaining_concept_accuracy"])

    def test_rejects_duplicate_seed_and_changed_training_protocol(self):
        runs = copy.deepcopy(self.runs)
        runs[1]["hyperparameters"]["seed"] = 42
        with self.assertRaisesRegex(ValueError, "Duplicate seed"):
            summarize_interventions(runs)
        runs = copy.deepcopy(self.runs)
        runs[1]["hyperparameters"]["protocol"] = "different"
        with self.assertRaisesRegex(ValueError, "Training protocols differ"):
            summarize_interventions(runs)

    def test_rejects_incomplete_or_tampered_predictions(self):
        runs = copy.deepcopy(self.runs)
        runs[1]["subsets"].pop()
        with self.assertRaisesRegex(ValueError, "all intervention subsets"):
            summarize_interventions(runs)
        runs = copy.deepcopy(self.runs)
        runs[1]["subsets"][0]["y_pred"][0] = 0
        with self.assertRaisesRegex(ValueError, "frozen threshold"):
            summarize_interventions(runs)

    def test_rejects_wrong_case_transition_counts(self):
        runs = copy.deepcopy(self.runs)
        runs[0]["subsets"][-1]["improved_case_ids"] = []
        with self.assertRaisesRegex(ValueError, "Case transitions"):
            summarize_interventions(runs)


if __name__ == "__main__":
    unittest.main()
