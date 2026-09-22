import os
import sys
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.run_m1 import _default_paths, _manifest_hash, evaluate_m1_test, select_validation_threshold
from src.dataset import get_dataloaders
from src.metrics import compute_metrics
from src.models import BlackBoxClassifier, load_blackbox_state_dict
from src.transforms import get_transforms


class M1ProtocolTests(unittest.TestCase):
    def test_efficientnet_has_one_feature_tree_and_forward_shape(self):
        model = BlackBoxClassifier(pretrained=False).eval()
        state_keys = model.state_dict().keys()
        self.assertFalse(any(key.startswith("backbone.") for key in state_keys))
        with torch.no_grad():
            image = torch.zeros(1, 3, 224, 224)
            self.assertEqual(tuple(model(image).shape), (1, 2))
            self.assertEqual(tuple(model.extract_features(image).shape), (1, 1280))

    def test_legacy_duplicate_keys_can_be_loaded(self):
        model = BlackBoxClassifier(pretrained=False)
        state = model.state_dict()
        legacy = dict(state)
        feature_key = next(key for key in state if key.startswith("features."))
        legacy["backbone.features.0.0.weight"] = state[feature_key].clone()
        load_blackbox_state_dict(model, legacy)

    def test_comparison_augmentation_and_eval_preprocessing(self):
        train_transform = get_transforms("train", augmentation_preset="comparison")
        names = [type(step).__name__ for step in train_transform.transforms]
        self.assertEqual(
            names[:5],
            ["RandomResizedCrop", "RandomHorizontalFlip", "RandomVerticalFlip", "ColorJitter", "RandomRotation"],
        )
        image = Image.new("RGB", (300, 250), (120, 90, 50))
        self.assertEqual(tuple(train_transform(image).shape), (3, 224, 224))
        valid_transform = get_transforms("valid", augmentation_preset="comparison")
        self.assertEqual(type(valid_transform.transforms[0]).__name__, "LetterboxResize")
        self.assertTrue(torch.equal(valid_transform(image), valid_transform(image)))

    def test_pr_auc_is_average_precision(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.6, 0.4, 0.8])
        metrics = compute_metrics(y_true, (y_prob >= 0.5).astype(int), y_prob)
        self.assertAlmostEqual(metrics["pr_auc"], average_precision_score(y_true, y_prob))

    def test_threshold_is_selected_from_validation(self):
        threshold, bacc = select_validation_threshold(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.55, 0.45, 0.8])
        )
        self.assertAlmostEqual(threshold, 0.45)
        self.assertAlmostEqual(bacc, 0.75)

    def test_train_dataloaders_exclude_test(self):
        bundle = get_dataloaders(
            manifest_path=os.path.join(PROJECT_ROOT, "data", "manifest.csv"),
            project_root=PROJECT_ROOT,
            batch_size=16,
            num_workers=0,
            augmentation_preset="comparison",
            include_test=False,
        )
        self.assertEqual(len(bundle["datasets"]["train"]), 413)
        self.assertEqual(len(bundle["datasets"]["valid"]), 203)
        self.assertNotIn("test", bundle)
        self.assertNotIn("test", bundle["datasets"])

    def test_default_paths_are_seed_specific(self):
        self.assertNotEqual(_default_paths(42, "comparison"), _default_paths(123, "comparison"))
        self.assertNotEqual(_default_paths(42, "comparison"), _default_paths(42, "legacy_letterbox"))

    def test_explicit_test_mode_uses_frozen_threshold(self):
        manifest = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
        model = BlackBoxClassifier(pretrained=False)
        config = {
            "seed": 7, "augmentation_preset": "comparison", "target_size": 224,
            "batch_size": 16, "class_weights": [0.64, 2.29],
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
        metrics = compute_metrics(np.array([0] * 294 + [1] * 101), preds, probs)
        with tempfile.TemporaryDirectory(prefix="m1_test_mode_") as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "model.pth")
            result_path = os.path.join(temp_dir, "test.json")
            torch.save(checkpoint, checkpoint_path)
            with patch("experiments.run_m1.Derm7ptDataset", FakeTestDataset), patch(
                "experiments.run_m1.evaluate", return_value=(metrics, 0.1, preds, probs)
            ) as mocked_evaluate:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = evaluate_m1_test(checkpoint_path, manifest, "cpu", result_path)
            self.assertEqual(result["decision_threshold"], 0.42)
            self.assertEqual(result["mode"], "final_test")
            self.assertEqual(len(result["test_predictions"]), 395)
            self.assertEqual(mocked_evaluate.call_args.kwargs["threshold"], 0.42)


if __name__ == "__main__":
    unittest.main()
