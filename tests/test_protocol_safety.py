"""Regression checks for frozen schemas, exports, interventions and seed summaries."""
import contextlib
import copy
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.dataset import CONCEPT_NAMES, CONCEPT_NUM_CLASSES, compute_concept_statistics, compute_diagnosis_weights
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.protocol import (manifest_hash, manifest_fingerprint, load_concept_schema,
                          validate_manifest, validate_concept_schema, save_experiment_results)
from src.metrics import compute_metrics, compute_concept_metrics
from src.intervention import evaluate_soft_interventions
from experiments.run_m3 import (StateWeightedCE, _default_paths, run_m3_experiment, evaluate_m3_test,
                                run_m3_intervention, validate_m3_config, M3_PROTOCOL)
from experiments.run_m2_lr import run_m2_experiment, evaluate_m2_test
from experiments.run_m2_mlp import run_m2_mlp_experiment
from experiments import run_m1, run_m2_lr, run_m2_mlp, run_m3, run_m4_st, run_m4_sg
from experiments.summarize_seeds import summarize_seeds

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifest.csv"


class TinyCBM(nn.Module):
    """A small differentiable stand-in, so protocol tests require no weights download."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.concept_heads = nn.ModuleDict({c: nn.Linear(3, k) for c, k in CONCEPT_NUM_CLASSES.items()})
        self.diagnosis_head = nn.Linear(28, 2)

    def forward(self, images):
        logits = {c: self.concept_heads[c](images) for c in CONCEPT_NAMES}
        vector = torch.cat([logits[c].softmax(-1) for c in CONCEPT_NAMES], dim=1)
        return self.diagnosis_head(vector), logits, vector


class TinyClassifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.head = nn.Linear(3, 2)

    def forward(self, images):
        return self.head(images)


class TinyDataset(Dataset):
    def __init__(self, manifest_path, split, **kwargs):
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df.split == split].reset_index(drop=True)
        self.label_mapping = load_concept_schema(manifest_path)["label_mapping"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return {"image": torch.tensor([index / 1000, 1., -1.]),
                "label": torch.tensor(int(row.diagnosis_binary)),
                "concept_indices": torch.tensor([self.label_mapping[c][row[c]] for c in CONCEPT_NAMES])}


class ProtocolSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="protocol_safety_")
        self.path = Path(self.temp.name)
        self.manifest = self.path / "manifest.csv"
        shutil.copyfile(MANIFEST, self.manifest)
        shutil.copyfile(ROOT / "data/label_mapping.json", self.path / "label_mapping.json")
        self.schema = load_concept_schema(self.manifest)
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)
        self.temp.cleanup()

    def swap_mapping(self):
        mapping = copy.deepcopy(self.schema["label_mapping"])
        mapping["pigment_network"]["typical"], mapping["pigment_network"]["atypical"] = (
            mapping["pigment_network"]["atypical"], mapping["pigment_network"]["typical"])
        (self.path / "label_mapping.json").write_text(json.dumps(mapping))

    def test_direct_training_apis_reject_aliased_outputs_before_data_loading(self):
        # Notebook users call these functions without the CLI preflight. Even
        # --overwrite must not allow a JSON export to replace a checkpoint.
        alias = self.path / "checkpoint.pth"
        original = b"existing checkpoint must be preserved"
        alias.write_bytes(original)
        for runner in [run_m1.run_m1_experiment, run_m2_experiment,
                       run_m2_mlp_experiment, run_m3_experiment, run_m4_st.run_m4_experiment, run_m4_sg.run_m4_sg_experiment]:
            for save_results in [True, False]:
                with self.subTest(runner=runner.__name__, save_results=save_results):
                    with self.assertRaisesRegex(ValueError, "distinct paths"):
                        runner(manifest_path=str(self.path / "missing.csv"),
                               checkpoint_path=str(alias),
                               results_path=str(self.path / "nested" / ".." / alias.name),
                               save_results=save_results, overwrite=True)
                    self.assertEqual(alias.read_bytes(), original)

    def test_f1_checkpoint_ranking_differs_from_bacc_and_ties_keep_first_epoch(self):
        # Dùng các dự đoán hợp lệ mà F1 và BAcc xếp hạng NGƯỢC nhau.
        # Ba epochs có dự đoán A/B/A: F1 phải chọn epoch 1 (không phải 2 hoặc 3).
        from test_m4_st_protocol import TinyHardCBM
        from test_m4_sg_protocol import TinyHardStopGradientCBM
        from experiments import m4_common
        train, valid = TinyDataset(str(self.manifest), "train"), TinyDataset(str(self.manifest), "valid")
        bundle = {"train": DataLoader(train, batch_size=128), "valid": DataLoader(valid, batch_size=128),
                  "datasets": {"train": train, "valid": valid},
                  "class_weights": compute_diagnosis_weights(str(self.manifest), str(ROOT))}
        truth = valid.df.diagnosis_binary.to_numpy(dtype=int)
        positives, negatives = np.flatnonzero(truth == 1), np.flatnonzero(truth == 0)

        def probability(tp, tn):
            prediction = np.ones(len(truth), dtype=int)
            prediction[positives] = 0
            prediction[positives[:tp]] = 1
            prediction[negatives[:tn]] = 0
            return np.where(prediction, .9, .1)

        a, b = probability(15, 135), probability(49, 64)
        metrics_a, metrics_b = [compute_metrics(truth, p >= .5, p) for p in [a, b]]
        self.assertGreater(metrics_a["f1_macro"], metrics_b["f1_macro"])
        self.assertLess(metrics_a["balanced_accuracy"], metrics_b["balanced_accuracy"])
        for label in ["m1", "m2_mlp", "m3", "m4_st", "m4_sg"]:
            for metric, epoch in [("f1_macro", 1), ("balanced_accuracy", 2)]:
                with self.subTest(model=label, criterion=metric), contextlib.ExitStack() as stack:
                    checkpoint = self.path / f"{label}_{metric}.pth"
                    sequence = iter([a, b, a, a if metric == "f1_macro" else b])
                    kwargs = dict(manifest_path=str(self.manifest), epochs=3, batch_size=128,
                                  device_name="cpu", checkpoint_path=str(checkpoint), save_results=False)
                    # F1 kiểm tra DEFAULT mới; BAcc kiểm tra chế độ tái lập protocol cũ.
                    if metric != "f1_macro":
                        kwargs["checkpoint_metric"] = metric
                    if label == "m2_mlp":
                        stack.enter_context(patch.object(run_m2_mlp, "predict_probabilities",
                                                        side_effect=lambda *args: next(sequence)))
                        runner = run_m2_mlp.run_m2_mlp_experiment
                    else:
                        module = run_m1 if label == "m1" else run_m3 if label == "m3" else m4_common
                        original = module.evaluate

                        def evaluate(*args, **kw):
                            result = original(*args, **kw)
                            p = next(sequence)
                            metrics = compute_metrics(truth, p >= .5, p)
                            if label.startswith("m4"):
                                return {**result, "diagnosis_metrics": metrics,
                                        "predictions": (p >= .5).astype(int), "probabilities": p}
                            if label == "m1":
                                return metrics, result[1], (p >= .5).astype(int), p
                            return (metrics, *result[1:5], (p >= .5).astype(int), p)

                        stack.enter_context(patch.object(module, "evaluate", side_effect=evaluate))
                        stack.enter_context(patch.object(module, "get_dataloaders", return_value=bundle))
                        if label == "m1":
                            stack.enter_context(patch.object(module, "BlackBoxClassifier", TinyClassifier))
                            stack.enter_context(patch.object(module, "load_blackbox_state_dict",
                                                            side_effect=lambda m, state: m.load_state_dict(state)))
                            runner = run_m1.run_m1_experiment
                        elif label == "m3":
                            stack.enter_context(patch.object(module, "get_soft_joint_cbm", TinyCBM))
                            stack.enter_context(patch.object(module, "load_soft_joint_cbm_state_dict",
                                                            side_effect=lambda m, state: m.load_state_dict(state)))
                            runner = run_m3.run_m3_experiment
                        else:
                            name, model = ("get_hard_joint_cbm", TinyHardCBM) if label == "m4_st" else (
                                "get_hard_stop_gradient_cbm", TinyHardStopGradientCBM)
                            stack.enter_context(patch.object(module, name, model))
                            kwargs["gradient_mode"] = "straight_through" if label == "m4_st" else "stop_gradient"
                            runner = m4_common.run_m4_experiment
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    result = runner(**kwargs)
                    frozen = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    self.assertEqual(result["best_epoch"], epoch)
                    self.assertEqual(frozen["epoch"], epoch)
                    self.assertEqual(result["hyperparameters"]["checkpoint_selection"], f"validation_{metric}_at_0.5")
                    self.assertEqual(result["hyperparameters"]["threshold_selection"],
                                     "maximize_validation_balanced_accuracy_tie_nearest_0.5")
                    self.assertAlmostEqual(result["history"][0]["validation_f1_macro_at_0.5"], metrics_a["f1_macro"])
                    self.assertAlmostEqual(result["history"][0]["validation_balanced_accuracy_at_0.5"],
                                           metrics_a["balanced_accuracy"])

    def test_checkpoint_criteria_have_separate_paths_and_legacy_config_is_recognized(self):
        from src.selection import metric_from_config
        from experiments import m4_common
        for paths in [run_m1._default_paths, run_m2_lr._default_paths, run_m2_mlp.default_paths,
                      run_m3._default_paths, m4_common._default_paths]:
            new = paths(checkpoint_metric="f1_macro")
            old = paths(checkpoint_metric="balanced_accuracy")
            self.assertNotEqual(new, old)
            self.assertIn("ckptf1macro", new[0])
            self.assertNotIn("ckptf1macro", old[0])
        self.assertEqual(metric_from_config({}), "balanced_accuracy")
        self.assertEqual(metric_from_config({"checkpoint_selection": "validation_balanced_accuracy_at_0.5"}),
                         "balanced_accuracy")
        for config in [{"checkpoint_selection": "test_f1_macro"},
                       {"checkpoint_selection": "validation_f1_macro_at_0.5", "checkpoint_metric": "balanced_accuracy"}]:
            with self.assertRaises(ValueError):
                metric_from_config(config)

    def test_new_and_legacy_hashes_allow_only_newline_conversion(self):
        raw = self.manifest.read_bytes().replace(b"\r\n", b"\n")
        self.manifest.write_bytes(raw)
        new = {"manifest_fingerprint": manifest_fingerprint(self.manifest)}
        old = {"manifest_sha256": manifest_hash(self.manifest)}
        self.manifest.write_bytes(raw.replace(b"\n", b"\r\n"))
        validate_manifest(new, self.manifest)
        validate_manifest(old, self.manifest)
        self.manifest.write_bytes(raw + b"\n")
        for checkpoint in [new, old]:
            with self.assertRaises(ValueError):
                validate_manifest(checkpoint, self.manifest)

    def test_schema_and_m2_inference_reject_changed_mapping(self):
        checkpoint = self.path / "m2.joblib"
        with contextlib.redirect_stdout(io.StringIO()):
            run_m2_experiment(str(self.manifest), c_grid=[0.1], checkpoint_path=str(checkpoint), save_results=False)
            baseline = evaluate_m2_test(str(checkpoint), str(self.manifest), save_results=False)
        self.assertEqual(len(baseline["test_predictions"]), 395)
        self.swap_mapping()
        with self.assertRaisesRegex(ValueError, "schema differs"):
            evaluate_m2_test(str(checkpoint), str(self.manifest), save_results=False)
        with self.assertRaises(ValueError):
            validate_concept_schema({"concept_schema": self.schema}, self.manifest)
        with self.assertRaisesRegex(ValueError, "no frozen"):
            validate_concept_schema({}, self.manifest)

    def test_weighted_loss_handles_unseen_states_and_weights_use_train_only(self):
        logits = torch.tensor([[1., 2., 3.], [0., 1., 0.]], requires_grad=True)
        criterion = StateWeightedCE([2., 1., 0.])
        zero = criterion(logits, torch.tensor([2, 2]))
        self.assertEqual(zero.item(), 0.)
        zero.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        targets = torch.tensor([0, 1])
        expected = nn.functional.cross_entropy(logits, targets, weight=torch.tensor([2., 1., 0.]))
        self.assertAlmostEqual(criterion(logits, targets).item(), expected.item())
        original = compute_concept_statistics(str(self.manifest), str(ROOT))
        df = pd.read_csv(self.manifest)
        for c in CONCEPT_NAMES:
            df.loc[df.split != "train", c] = next(iter(self.schema["label_mapping"][c]))
        df.to_csv(self.manifest, index=False)
        self.assertEqual(original, compute_concept_statistics(str(self.manifest), str(ROOT)))
        self.assertNotIn("unweighted", _default_paths(42, "legacy_letterbox", 1.)[0])

    def test_final_m3_protocol_rejects_old_or_unweighted_checkpoint(self):
        weights = {c: [1.] * k for c, k in CONCEPT_NUM_CLASSES.items()}
        config = {"protocol": M3_PROTOCOL, "concept_weighting": "balanced", "concept_state_weights": weights}
        self.assertEqual(set(validate_m3_config(config)), set(CONCEPT_NAMES))
        for changed in [{**config, "concept_weighting": "unweighted"},
                        {**config, "protocol": "old"}, {**config, "concept_state_weights": None}]:
            checkpoint = {"config": changed, "decision_threshold": .5, "concept_schema": self.schema,
                          "manifest_fingerprint": manifest_fingerprint(self.manifest)}
            path = self.path / "old.pth"
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                evaluate_m3_test(str(path), str(self.manifest), "cpu", save_results=False)
            with self.assertRaises(ValueError):
                run_m3_intervention(str(path), str(self.path / "missing.json"), str(self.manifest))

    def test_profile_exact_match_and_per_state_f1(self):
        result = compute_concept_metrics({"a": [0, 1, 1], "b": [0, 0, 1]},
                                         {"a": [0, 0, 1], "b": [0, 1, 1]}, {"a": 3, "b": 2})
        self.assertAlmostEqual(result["exact_match_accuracy"], 2/3)
        np.testing.assert_allclose(result["per_concept"]["a"]["per_state_f1"], [2/3, 2/3, 0])
        self.assertEqual(result["per_concept"]["a"]["support"], [1, 2, 0])

    def test_all_intervention_subsets_keep_other_groups_and_frozen_threshold(self):
        class RecordingHead(nn.Linear):
            def __init__(self):
                super().__init__(28, 2)
                self.inputs = []

            def forward(self, x):
                self.inputs.append(x.clone())
                return super().forward(x)

        head = RecordingHead()
        with torch.no_grad():
            head.weight.zero_()
            head.bias.copy_(torch.tensor([0., 1.]))  # probability .731 < frozen .8
        probs = np.concatenate([np.full((2, k), 1/k) for k in CONCEPT_NUM_CLASSES.values()], axis=1)
        targets = np.zeros((2, 7), dtype=int)
        result = evaluate_soft_interventions(head, probs, targets, [0, 1], self.schema, .8, [10, 20])
        self.assertEqual(result["num_subsets"], 128)
        self.assertEqual(len({tuple(row["groups"]) for row in result["subsets"]}), 128)
        self.assertEqual([row["num_subsets"] for row in result["curve"]], [1, 7, 21, 35, 35, 21, 7, 1])
        self.assertEqual(result["baseline_y_pred"], [0, 0])
        self.assertIsNone(result["curve"][-1]["remaining_concept_accuracy"])
        self.assertAlmostEqual(result["intervention_curve_auc"]["balanced_accuracy"]["absolute"], .5)
        self.assertAlmostEqual(result["intervention_curve_auc"]["balanced_accuracy"]["gain_over_baseline"], 0.)
        for row, seen in zip(result["subsets"], head.inputs[1:]):
            self.assertEqual(row["y_pred"], [0, 0])
            for c in CONCEPT_NAMES:
                start, k = self.schema["offsets"][c], CONCEPT_NUM_CLASSES[c]
                expected = np.zeros((2, k)) if c in row["groups"] else probs[:, start:start+k]
                if c in row["groups"]:
                    expected[:, 0] = 1.
                np.testing.assert_allclose(seen[:, start:start+k].numpy(), expected)

    def test_m3_weighted_train_test_export_and_intervention_pipeline(self):
        train, valid = TinyDataset(str(self.manifest), "train"), TinyDataset(str(self.manifest), "valid")
        bundle = {"train": DataLoader(train, batch_size=128), "valid": DataLoader(valid, batch_size=128),
                  "datasets": {"train": train, "valid": valid},
                  "class_weights": compute_diagnosis_weights(str(self.manifest), str(ROOT))}
        checkpoint, predictions = str(self.path / "m3.pth"), str(self.path / "test.json")
        with patch("experiments.run_m3.get_dataloaders", return_value=bundle) as loader, \
             patch("experiments.run_m3.get_soft_joint_cbm", TinyCBM), \
             patch("experiments.run_m3.load_soft_joint_cbm_state_dict", lambda m, s: m.load_state_dict(s)), \
             patch("experiments.run_m3.Derm7ptDataset", TinyDataset), contextlib.redirect_stdout(io.StringIO()):
            run_m3.main(["--manifest_path", str(self.manifest), "--epochs", "1",
                         "--device", "cpu", "--num_workers", "0", "--checkpoint_path", checkpoint,
                         "--results_path", predictions])
            self.assertFalse(loader.call_args.kwargs["include_test"])
            validation = test = json.loads(Path(predictions).read_text())
            self.assertEqual(test["mode"], "train_validation_test")
        intervention_path = self.path / "intervention.json"
        completed = subprocess.run(
            [sys.executable, str(ROOT / "experiments/run_m3.py"), "--mode", "intervention",
             "--checkpoint_path", checkpoint, "--predictions_path", predictions,
             "--manifest_path", str(self.manifest), "--results_path", str(intervention_path)],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"})
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(intervention_path.read_text())
        self.assertEqual(validation["hyperparameters"]["concept_weighting"], "balanced")
        self.assertEqual(validation["hyperparameters"]["protocol"], M3_PROTOCOL)
        self.assertEqual(len(validation["validation_predictions"]), 203)
        self.assertEqual(len(test["test_predictions"]), 395)
        self.assertEqual(result["num_subsets"], 128)
        np.testing.assert_allclose(result["baseline_y_prob"], [row["y_prob"] for row in test["test_predictions"]], atol=1e-6)
        self.assertEqual(test["checkpoint_sha256"], manifest_hash(checkpoint))
        for row in test["test_predictions"]:
            for c in CONCEPT_NAMES:
                self.assertEqual(len(row["concept_probabilities"][c]), CONCEPT_NUM_CLASSES[c])
                self.assertAlmostEqual(sum(row["concept_probabilities"][c]), 1., places=5)
                self.assertEqual(row["concept_pred"][c], int(np.argmax(row["concept_probabilities"][c])))
        test["checkpoint_sha256"] = "different"
        Path(predictions).write_text(json.dumps(test))
        with self.assertRaisesRegex(ValueError, "does not belong"):
            run_m3_intervention(checkpoint, predictions, str(self.manifest), str(self.path / "bad.json"))
        self.assertFalse((self.path / "bad.json").exists())

    def test_m1_combined_command_reloads_frozen_checkpoint(self):
        train, valid = TinyDataset(str(self.manifest), "train"), TinyDataset(str(self.manifest), "valid")
        bundle = {"train": DataLoader(train, batch_size=128), "valid": DataLoader(valid, batch_size=128),
                  "datasets": {"train": train, "valid": valid},
                  "class_weights": compute_diagnosis_weights(str(self.manifest), str(ROOT))}
        checkpoint = str(self.path / "m1.pth")
        with patch("experiments.run_m1.get_dataloaders", return_value=bundle) as loader, \
             patch("experiments.run_m1.BlackBoxClassifier", TinyClassifier), \
             patch("experiments.run_m1.load_blackbox_state_dict", lambda m, s: m.load_state_dict(s)), \
             patch("experiments.run_m1.Derm7ptDataset", TinyDataset), contextlib.redirect_stdout(io.StringIO()):
            run_m1.main(["--manifest_path", str(self.manifest), "--epochs", "1",
                         "--device", "cpu", "--num_workers", "0", "--checkpoint_path", checkpoint,
                         "--results_path", str(self.path / "m1_results.json")])
        validation = test = json.loads((self.path / "m1_results.json").read_text())
        self.assertEqual(test["mode"], "train_validation_test")
        frozen = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertFalse(loader.call_args.kwargs["include_test"])
        self.assertEqual(test["decision_threshold"], frozen["decision_threshold"])
        self.assertEqual(test["best_epoch"], validation["best_epoch"])
        self.assertEqual(len(test["test_predictions"]), 395)

    def test_lr_combined_command_and_existing_test_preflight(self):
        checkpoint = self.path / "lr.joblib"
        args = ["--manifest_path", str(self.manifest), "--checkpoint_path", str(checkpoint),
                "--results_path", str(self.path / "lr_results.json")]
        with contextlib.redirect_stdout(io.StringIO()):
            run_m2_lr.main(args)
        validation = test = json.loads((self.path / "lr_results.json").read_text())
        self.assertEqual(test["mode"], "train_validation_test")
        self.assertEqual(test["decision_threshold"], validation["decision_threshold"])
        self.assertEqual(len(test["test_predictions"]), 395)
        with contextlib.redirect_stdout(io.StringIO()):
            run_m2_lr.main(args + ["--mode", "test", "--overwrite"])
        reexport = json.loads((self.path / "lr_results.json").read_text())
        self.assertEqual(reexport["validation_predictions"], validation["validation_predictions"])
        self.assertEqual(reexport["validation_metrics"], validation["validation_metrics"])
        self.assertEqual(reexport["test_predictions"], test["test_predictions"])
        before = (self.path / "lr_results.json").read_bytes()
        different = copy.deepcopy(test)
        different["decision_threshold"] = .123
        with self.assertRaisesRegex(ValueError, "decision_threshold differs"):
            save_experiment_results(self.path / "lr_results.json", test=different, overwrite=True)
        self.assertEqual((self.path / "lr_results.json").read_bytes(), before)
        # An existing combined JSON must stop every CLI before training.
        for module in [run_m1, run_m2_lr, run_m2_mlp, run_m3, run_m4_st, run_m4_sg]:
            fresh = self.path / (module.__name__.split(".")[-1] + "_new.pth")
            with self.subTest(module=module.__name__), self.assertRaises(FileExistsError):
                module.main(["--checkpoint_path", str(fresh),
                             "--results_path", str(self.path / "lr_results.json")])
            self.assertFalse(fresh.exists())
        with self.assertRaisesRegex(ValueError, "distinct paths"):
            run_m2_lr.main(["--checkpoint_path", str(checkpoint), "--results_path", str(checkpoint), "--overwrite"])

    def test_seed_summary_checks_protocol_distinct_seeds_and_metric_integrity(self):
        df = pd.read_csv(self.manifest)
        test = df[df.split == "test"]
        labels = test.diagnosis_binary.to_numpy(dtype=int)
        paths, runs = [], []
        for i, seed in enumerate([42, 123, 2026]):
            probs = np.where(labels == 1, .9, .1)
            probs[:i*10] = 1-probs[:i*10]
            preds = (probs >= .5).astype(int)
            run = {"model": "M1_BlackBox_EfficientNetB0", "mode": "final_test",
                   "hyperparameters": {"seed": seed, "epochs": 10, "device": "cpu"},
                   "manifest_fingerprint": manifest_fingerprint(self.manifest), "decision_threshold": .5,
                   "test_metrics": compute_metrics(labels, preds, probs),
                   "test_predictions": [{"case_num": int(case), "y_true": int(y), "y_pred": int(p), "y_prob": float(s)}
                                        for case, y, p, s in zip(test.case_num, labels, preds, probs)]}
            path = self.path / f"seed{seed}.json"
            path.write_text(json.dumps(run))
            paths.append(path)
            runs.append(run)
        summary = summarize_seeds(paths, self.manifest, self.path / "summary.json")
        values = [run["test_metrics"]["balanced_accuracy"] for run in runs]
        self.assertAlmostEqual(summary["diagnosis_metrics"]["balanced_accuracy"]["mean"], np.mean(values))
        self.assertAlmostEqual(summary["diagnosis_metrics"]["balanced_accuracy"]["sample_sd"], np.std(values, ddof=1))
        for mutation in ["duplicate_seed", "changed_protocol", "bad_metrics", "wrong_cases"]:
            changed = copy.deepcopy(runs[-1])
            if mutation == "duplicate_seed":
                changed["hyperparameters"]["seed"] = 42
            elif mutation == "changed_protocol":
                changed["hyperparameters"]["epochs"] = 20
            elif mutation == "bad_metrics":
                changed["test_metrics"]["balanced_accuracy"] = .123
            else:
                changed["test_predictions"] = changed["test_predictions"][1:]
            paths[-1].write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                summarize_seeds(paths, self.manifest, self.path / "bad_summary.json")
        self.assertFalse((self.path / "bad_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
