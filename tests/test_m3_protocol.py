import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Giữ đúng thứ tự import an toàn: src.dataset (kéo theo torch) trước sklearn.
from src.dataset import CONCEPT_NAMES, CONCEPT_NUM_CLASSES  # noqa: F401

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from experiments.run_m3 import (
    _concept_loss,
    _default_paths,
    _manifest_hash,
    compute_train_observed_states,
    evaluate_m3_test,
    select_validation_threshold,
)
from src.metrics import compute_metrics
from src.models import SoftJointCBM, load_soft_joint_cbm_state_dict


class M3ProtocolTests(unittest.TestCase):
    def test_soft_joint_cbm_forward_shapes_and_soft_bottleneck(self):
        model = SoftJointCBM(CONCEPT_NAMES, CONCEPT_NUM_CLASSES, pretrained=False).eval()
        with torch.no_grad():
            image = torch.zeros(2, 3, 224, 224)
            diag_logits, concept_logits, concept_vector = model(image)

            self.assertEqual(tuple(diag_logits.shape), (2, 2))
            self.assertEqual(tuple(concept_vector.shape), (2, 28))
            for name in CONCEPT_NAMES:
                self.assertEqual(tuple(concept_logits[name].shape), (2, CONCEPT_NUM_CLASSES[name]))

            # Mỗi nhóm concept là một softmax độc lập (tổng = 1); 7 nhóm -> tổng vector = 7
            self.assertTrue(torch.allclose(concept_vector.sum(dim=-1), torch.full((2,), 7.0), atol=1e-4))

    def test_diagnosis_head_is_pure_linear_on_28d_bottleneck(self):
        # g phải là Linear thuần (không hidden layer) để đối chiếu công bằng với M2 (Logistic Regression)
        model = SoftJointCBM(CONCEPT_NAMES, CONCEPT_NUM_CLASSES, pretrained=False)
        self.assertIsInstance(model.diagnosis_head, nn.Linear)
        self.assertEqual(model.diagnosis_head.in_features, 28)
        self.assertEqual(model.diagnosis_head.out_features, 2)

    def test_state_dict_round_trip(self):
        model = SoftJointCBM(CONCEPT_NAMES, CONCEPT_NUM_CLASSES, pretrained=False)
        state = model.state_dict()
        clone = SoftJointCBM(CONCEPT_NAMES, CONCEPT_NUM_CLASSES, pretrained=False)
        load_soft_joint_cbm_state_dict(clone, state)

    def test_concept_loss_is_unweighted_mean_of_seven_heads(self):
        torch.manual_seed(0)
        batch = 4
        concept_logits = {
            name: torch.randn(batch, CONCEPT_NUM_CLASSES[name]) for name in CONCEPT_NAMES
        }
        concept_indices = torch.stack([
            torch.randint(0, CONCEPT_NUM_CLASSES[name], (batch,)) for name in CONCEPT_NAMES
        ], dim=1)
        criterion = nn.CrossEntropyLoss()

        expected = torch.stack([
            criterion(concept_logits[name], concept_indices[:, i]) for i, name in enumerate(CONCEPT_NAMES)
        ]).mean()
        actual = _concept_loss(concept_logits, concept_indices, criterion)
        self.assertAlmostEqual(actual.item(), expected.item(), places=6)

    def test_train_observed_states_restricted_to_train_split(self):
        df = pd.DataFrame({
            "pigment_network": ["absent", "absent", "typical"],
            "streaks": ["absent", "absent", "absent"],
            "pigmentation": ["absent", "absent", "absent"],
            "regression_structures": ["absent", "absent", "absent"],
            "dots_and_globules": ["absent", "absent", "absent"],
            "blue_whitish_veil": ["absent", "absent", "absent"],
            "vascular_structures": ["absent", "absent", "absent"],
        })
        label_mapping = {
            "pigment_network": {"absent": 0, "atypical": 1, "typical": 2},
            "streaks": {"absent": 0, "irregular": 1, "regular": 2},
            "pigmentation": {"absent": 0, "diffuse irregular": 1, "diffuse regular": 2, "localized irregular": 3, "localized regular": 4},
            "regression_structures": {"absent": 0, "blue areas": 1, "combinations": 2, "white areas": 3},
            "dots_and_globules": {"absent": 0, "irregular": 1, "regular": 2},
            "blue_whitish_veil": {"absent": 0, "present": 1},
            "vascular_structures": {"absent": 0, "arborizing": 1, "comma": 2, "dotted": 3, "hairpin": 4, "linear irregular": 5, "within regression": 6, "wreath": 7},
        }
        observed = compute_train_observed_states(df, label_mapping)
        self.assertEqual(observed["pigment_network"], [0, 2])  # absent, typical - KHÔNG có atypical (=1)
        self.assertEqual(observed["blue_whitish_veil"], [0])

    def test_default_paths_are_config_specific(self):
        self.assertNotEqual(_default_paths(42, "legacy_letterbox", 1.0), _default_paths(123, "legacy_letterbox", 1.0))
        self.assertNotEqual(_default_paths(42, "legacy_letterbox", 1.0), _default_paths(42, "comparison", 1.0))
        self.assertNotEqual(_default_paths(42, "legacy_letterbox", 1.0), _default_paths(42, "legacy_letterbox", 0.5))

    def test_threshold_selection_matches_expected_tie_break(self):
        threshold, bacc = select_validation_threshold(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.55, 0.45, 0.8])
        )
        self.assertAlmostEqual(threshold, 0.45)
        self.assertAlmostEqual(bacc, 0.75)

    def test_explicit_test_mode_uses_frozen_threshold_and_reports_concept_metrics(self):
        manifest = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
        model = SoftJointCBM(CONCEPT_NAMES, CONCEPT_NUM_CLASSES, pretrained=False)
        config = {
            "seed": 7, "augmentation_preset": "comparison", "target_size": 224,
            "batch_size": 16, "class_weights": [0.64, 2.29], "concept_loss_weight": 1.0,
        }
        checkpoint = {
            "epoch": 1, "model_state_dict": model.state_dict(), "config": config,
            "decision_threshold": 0.42, "manifest_sha256": _manifest_hash(manifest),
        }

        class FakeTestDataset:
            def __init__(self, *args, **kwargs):
                self.df = pd.DataFrame({
                    "case_num": range(1, 396), "source_index": range(395),
                    "split": ["test"] * 395, "diagnosis": ["sample"] * 395,
                    "is_inconsistent_profile": [False] * 395,
                    "diagnosis_binary": [0] * 294 + [1] * 101,
                })

            def __len__(self):
                return 395

            def __getitem__(self, index):
                raise AssertionError("The mocked evaluator should not read images")

        probs = np.array([0.1] * 294 + [0.9] * 101)
        preds = (probs >= 0.42).astype(int)
        diag_metrics = compute_metrics(np.array([0] * 294 + [1] * 101), preds, probs)
        concept_metrics = {"overall_concept_accuracy": 0.5, "overall_f1_macro_all_defined": 0.4, "per_concept": {}}

        with tempfile.TemporaryDirectory(prefix="m3_test_mode_") as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "model.pth")
            result_path = os.path.join(temp_dir, "test.json")
            torch.save(checkpoint, checkpoint_path)
            with patch("experiments.run_m3.Derm7ptDataset", FakeTestDataset), patch(
                "experiments.run_m3.evaluate",
                return_value=(diag_metrics, concept_metrics, 0.1, 0.05, 0.05, preds, probs),
            ) as mocked_evaluate:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = evaluate_m3_test(checkpoint_path, manifest, "cpu", result_path)
            self.assertEqual(result["decision_threshold"], 0.42)
            self.assertEqual(result["mode"], "final_test")
            self.assertEqual(len(result["test_predictions"]), 395)
            self.assertIn("test_concept_metrics", result)
            self.assertEqual(mocked_evaluate.call_args.kwargs["threshold"], 0.42)


if __name__ == "__main__":
    unittest.main()
