# Kiểm tra protocol M4-ST: hard forward, straight-through gradient và train/test/intervention.
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.dataset import CONCEPT_NAMES, CONCEPT_NUM_CLASSES, compute_diagnosis_weights
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.cbm import HardJointCBM, SoftJointCBM, hard_categorical, make_diagnosis_head
from src.intervention import evaluate_hard_interventions
from src.protocol import load_concept_schema, manifest_fingerprint
from experiments import (m4_common, m4_interventions_common,
                         run_m4_st as run_m4, run_m4_st_interventions as run_m4_interventions)
from experiments.run_m3_interventions import export_report, summarize_interventions
from experiments.summarize_seeds import summarize_seeds

ROOT = Path(__file__).resolve().parents[1]


class TinyHardCBM(HardJointCBM):
    """Exercise the production hard/ST forward with a small trainable backbone."""
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        self.concept_names = list(CONCEPT_NAMES)
        self.concept_num_classes = dict(CONCEPT_NUM_CLASSES)
        self.features = nn.Linear(3, 8)
        self.dropout = nn.Dropout(.2)
        self.concept_heads = nn.ModuleDict({c: nn.Linear(8, k) for c, k in CONCEPT_NUM_CLASSES.items()})
        self.diagnosis_head = make_diagnosis_head(28, architecture=kwargs.get("diagnosis_head", "linear"))


class TinyDataset(Dataset):
    def __init__(self, manifest_path, split, **kwargs):
        df = pd.read_csv(manifest_path)
        self.df = df[df.split == split].reset_index(drop=True)
        self.label_mapping = load_concept_schema(manifest_path)["label_mapping"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return {"image": torch.tensor([index/1000, 1., -1.]),
                "label": torch.tensor(int(row.diagnosis_binary)),
                "concept_indices": torch.tensor([self.label_mapping[c][row[c]] for c in CONCEPT_NAMES])}


class M4STProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m4_st_protocol_")
        self.path = Path(self.temp.name)
        self.manifest = self.path / "manifest.csv"
        shutil.copy2(ROOT / "data/manifest.csv", self.manifest)
        shutil.copy2(ROOT / "data/label_mapping.json", self.path / "label_mapping.json")
        self.schema = load_concept_schema(self.manifest)
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)
        self.temp.cleanup()

    @contextlib.contextmanager
    def pipeline(self):
        train, valid = TinyDataset(self.manifest, "train"), TinyDataset(self.manifest, "valid")
        bundle = {"train": DataLoader(train, batch_size=128), "valid": DataLoader(valid, batch_size=128),
                  "datasets": {"train": train, "valid": valid},
                  "class_weights": compute_diagnosis_weights(str(self.manifest), str(ROOT))}
        with patch.object(m4_common, "get_dataloaders", return_value=bundle) as loader, \
             patch.object(m4_common, "get_hard_joint_cbm", TinyHardCBM), \
             patch.object(m4_common, "Derm7ptDataset", TinyDataset), \
             contextlib.redirect_stdout(io.StringIO()):
            yield loader

    def args(self, seed=42):
        return ["--manifest_path", str(self.manifest), "--epochs", "1", "--batch_size", "128",
                "--device", "cpu", "--num_workers", "0", "--seed", str(seed),
                "--checkpoint_path", str(self.path / f"seed{seed}.pth"),
                "--results_path", str(self.path / f"seed{seed}.json")]

    def train(self, seed=42, extra=None):
        with self.pipeline() as loader:
            result = run_m4.main(self.args(seed) + (extra or []))
            self.assertFalse(loader.call_args.kwargs["include_test"])
        return result

    def test_hard_forward_is_exact_even_during_training_and_ties_are_deterministic(self):
        logits = torch.tensor([[0., 0., 0.], [-2., 4., 1.]], requires_grad=True)
        expected = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        for training in (True, False):
            self.assertTrue(torch.equal(hard_categorical(logits, training), expected))
        model = TinyHardCBM()
        inputs = []
        handle = model.diagnosis_head.register_forward_pre_hook(lambda m, args: inputs.append(args[0].detach()))
        for training in (True, False):
            model.train(training)
            _, logits, vector = model(torch.ones(4, 3))
            self.assertTrue(torch.equal(vector.detach(), inputs[-1]))
            self.assertTrue(torch.all((vector == 0) | (vector == 1)))
            self.assertTrue(torch.equal(vector.sum(-1), torch.full((4,), 7.)))
            for c in CONCEPT_NAMES:
                start, k = self.schema["offsets"][c], CONCEPT_NUM_CLASSES[c]
                self.assertTrue(torch.equal(vector[:, start:start+k].sum(-1), torch.ones(4)))
                self.assertTrue(torch.equal(vector[:, start:start+k].argmax(-1), logits[c].softmax(-1).argmax(-1)))
        handle.remove()

    def test_diagnosis_loss_alone_reaches_concept_heads_and_backbone(self):
        torch.manual_seed(7)
        model = TinyHardCBM().train()
        diagnosis, _, _ = model(torch.tensor([[.1, 1., -1.], [.3, -.4, .2], [-.7, .1, 1.]]))
        nn.functional.cross_entropy(diagnosis, torch.tensor([0, 1, 0])).backward()
        for parameter in [model.features.weight, *[model.concept_heads[c].weight for c in CONCEPT_NAMES]]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def test_production_model_keeps_m3_architecture_and_initialization(self):
        torch.manual_seed(42)
        soft = SoftJointCBM(CONCEPT_NAMES, CONCEPT_NUM_CLASSES, pretrained=False)
        torch.manual_seed(42)
        hard = HardJointCBM(CONCEPT_NAMES, CONCEPT_NUM_CLASSES, pretrained=False).eval()
        for name, value in soft.state_dict().items():
            self.assertTrue(torch.equal(value, hard.state_dict()[name]), name)
        self.assertIsInstance(hard.diagnosis_head, nn.Linear)
        with torch.no_grad():
            diagnosis, concepts, vector = hard(torch.zeros(2, 3, 224, 224))
        self.assertEqual(tuple(diagnosis.shape), (2, 2))
        self.assertEqual(tuple(vector.shape), (2, 28))
        self.assertTrue(torch.equal(vector.sum(-1), torch.full((2,), 7.)))
        self.assertEqual(set(concepts), set(CONCEPT_NAMES))

    def test_default_paths_separate_seeds_and_nondefault_configurations(self):
        base = run_m4._default_paths(42)
        self.assertNotEqual(base, run_m4._default_paths(123))
        self.assertNotEqual(base, run_m4._default_paths(42, "comparison"))
        self.assertNotEqual(base, run_m4._default_paths(42, concept_loss_weight=.5))
        self.assertIn("m4_hard_joint_cbm_seed42", base[0])

    def test_upgraded_pilot_paths_and_validation_only_export(self):
        upgraded = run_m4._default_paths(42, diagnosis_head="mlp128", diagnosis_lr=.001, epochs=20)
        self.assertIn("m4_hard_joint_cbm_mlp128_headlr0.001_epochs20_seed42_ckptf1macro", upgraded[0])
        self.assertNotEqual(upgraded, run_m4._default_paths(42))
        with patch.object(m4_common, "evaluate_m4_test", side_effect=AssertionError("Pilot must not read test")):
            exported = self.train(extra=["--diagnosis_head", "mlp128", "--diagnosis_lr", "0.001", "--skip_test"])
        self.assertEqual(exported["mode"], "train_validation")
        self.assertNotIn("test_predictions", exported)
        saved = json.loads((self.path / "seed42.json").read_text())
        checkpoint = torch.load(self.path / "seed42.pth", weights_only=True)
        self.assertEqual(saved["hyperparameters"], checkpoint["config"])
        self.assertEqual(saved["hyperparameters"]["diagnosis_head"], "MLP(28, 128, 2)")
        self.assertEqual(saved["hyperparameters"]["diagnosis_learning_rate"], .001)

    def test_mlp_intervention_preset_resolves_cohort_and_separate_output(self):
        # Resolve all three inputs from the MLP preset rather than explicit paths.
        with patch.object(m4_common, "ROOT", self.path), \
             patch.object(m4_interventions_common, "ROOT", self.path):
            for seed in (42, 123, 2026):
                checkpoint, result = run_m4._default_paths(seed, diagnosis_head="mlp128", diagnosis_lr=.001, epochs=1)
                self.train(seed, extra=["--diagnosis_head", "mlp128", "--diagnosis_lr", ".001",
                                       "--checkpoint_path", checkpoint, "--results_path", result])
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_m4_interventions.main(["--manifest_path", str(self.manifest),
                    "--diagnosis_head", "mlp128", "--diagnosis_lr", ".001", "--epochs", "1"])
            output = self.path / "results/f1_macro/m4_st/mlp128_e1_headlr0.001/intervention"
            self.assertTrue((output / "summary.json").exists())
            self.assertFalse((self.path / "results/f1_macro/m4_st/linear_e10/intervention").exists())
            self.assertEqual(result["training_protocol"]["diagnosis_head"], "MLP(28, 128, 2)")
            self.assertEqual(result["num_subsets_per_seed"], 128)

    def test_mlp_head_keeps_hard_inputs_and_st_gradient_flow(self):
        # Head MLP đổi classifier nhưng vẫn phải nhận one-hot và giữ gradient ST.
        torch.manual_seed(7)
        model = TinyHardCBM(diagnosis_head="mlp128").train()
        diagnosis, _, vector = model(torch.tensor([[.1, 1., -1.], [.3, -.4, .2], [-.7, .1, 1.]]))
        self.assertIsInstance(model.diagnosis_head[1], nn.LayerNorm)
        self.assertEqual(model.diagnosis_head[3].p, .3)
        self.assertTrue(torch.all((vector.detach() == 0) | (vector.detach() == 1)))
        nn.functional.cross_entropy(diagnosis, torch.tensor([0, 1, 0])).backward()
        for parameter in [model.features.weight, *[model.concept_heads[c].weight for c in CONCEPT_NAMES]]:
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def test_mlp_st_train_test_intervention_round_trip(self):
        # Kiểm tra checkpoint MLP dựng lại đúng head khi test và intervention.
        result = self.train(extra=["--diagnosis_head", "mlp128", "--diagnosis_lr", "0.001"])
        self.assertEqual(result["hyperparameters"]["diagnosis_head"], "MLP(28, 128, 2)")
        self.assertEqual(result["hyperparameters"]["diagnosis_normalization"], "LayerNorm")
        output = self.path / "mlp_st_intervention.json"
        run = run_m4.run_m4_intervention(str(self.path / "seed42.pth"),
            str(self.path / "seed42.json"), str(self.manifest), str(output))
        self.assertEqual(run["num_subsets"], 128)
        self.assertEqual(run["baseline_y_pred"], [row["y_pred"] for row in result["test_predictions"]])
        np.testing.assert_allclose(run["baseline_y_prob"],
            [row["y_prob"] for row in result["test_predictions"]], atol=1e-6, rtol=1e-5)

    def test_default_cli_trains_then_tests_with_actual_hard_exports(self):
        result = self.train()
        saved = json.loads((self.path / "seed42.json").read_text())
        self.assertEqual(result["mode"], "train_validation_test")
        self.assertEqual(saved["mode"], "train_validation_test")
        self.assertEqual(len(saved["validation_predictions"]), 203)
        self.assertEqual(len(saved["test_predictions"]), 395)
        checkpoint = torch.load(self.path / "seed42.pth", weights_only=True)
        self.assertEqual(saved["decision_threshold"], checkpoint["decision_threshold"])
        self.assertEqual(saved["hyperparameters"]["protocol"], run_m4.M4_PROTOCOL)
        self.assertEqual(saved["hyperparameters"]["concept_weighting"], "balanced")
        for row in saved["test_predictions"]:
            vector = np.asarray(row["hard_concept_vector"])
            self.assertEqual(vector.sum(), 7)
            self.assertEqual(row["y_pred"], int(row["y_prob"] >= saved["decision_threshold"]))
            for c in CONCEPT_NAMES:
                index = row["concept_pred"][c]
                self.assertEqual(index, int(np.argmax(row["concept_probabilities"][c])))
                start, k = self.schema["offsets"][c], CONCEPT_NUM_CLASSES[c]
                np.testing.assert_array_equal(vector[start:start+k], np.eye(k)[index])

    def test_skip_test_never_calls_evaluator_and_can_finish_test_later(self):
        with patch.object(m4_common, "evaluate_m4_test", side_effect=AssertionError("Pilot must not read test")):
            validation = self.train(extra=["--skip_test"])
        self.assertNotIn("test_predictions", validation)
        saved = json.loads((self.path / "seed42.json").read_text())
        self.assertEqual(saved["mode"], "train_validation")
        self.assertNotIn("test_metrics", saved)
        with self.pipeline():
            run_m4.main(self.args() + ["--mode", "test", "--overwrite"])
        combined = json.loads((self.path / "seed42.json").read_text())
        self.assertEqual(combined["mode"], "train_validation_test")
        self.assertEqual(combined["validation_predictions"], saved["validation_predictions"])
        self.assertEqual(combined["decision_threshold"], saved["decision_threshold"])

    def test_no_save_does_not_skip_test(self):
        result = self.train(extra=["--no_save"])
        self.assertIn("test_predictions", result)
        self.assertTrue((self.path / "seed42.pth").exists())
        self.assertFalse((self.path / "seed42.json").exists())

    def test_invalid_pilot_flag_is_rejected_before_reading_checkpoint(self):
        for mode in ("test", "intervention"):
            with self.subTest(mode=mode), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    run_m4.main(["--mode", mode, "--skip_test"])
                self.assertEqual(error.exception.code, 2)

    def test_output_aliases_and_manifest_overwrite_fail_before_loading_data(self):
        checkpoint = self.path / "preserve.pth"
        checkpoint.write_bytes(b"preserve")
        with patch.object(m4_common, "get_dataloaders", side_effect=AssertionError("No data loading")):
            for save in (True, False):
                with self.assertRaisesRegex(ValueError, "distinct paths"):
                    run_m4.run_m4_experiment(checkpoint_path=str(checkpoint),
                        results_path=str(self.path / "nested" / ".." / checkpoint.name),
                        overwrite=True, save_results=save)
            original = self.manifest.read_bytes()
            with self.assertRaisesRegex(ValueError, "overwrite its inputs"):
                run_m4.run_m4_experiment(manifest_path=str(self.manifest),
                    checkpoint_path=str(self.manifest), results_path=str(self.path / "other.json"), overwrite=True)
            self.assertEqual(self.manifest.read_bytes(), original)
        self.assertEqual(checkpoint.read_bytes(), b"preserve")

    def test_existing_results_fail_before_training(self):
        result = self.path / "seed42.json"
        result.write_text("existing")
        with patch.object(m4_common, "get_dataloaders", side_effect=AssertionError("No training")):
            with self.assertRaises(FileExistsError):
                run_m4.main(self.args())
        self.assertEqual(result.read_text(), "existing")
        self.assertFalse((self.path / "seed42.pth").exists())

    def test_checkpoint_identity_and_schema_are_enforced(self):
        self.train(extra=["--skip_test"])
        checkpoint = self.path / "seed42.pth"
        original = torch.load(checkpoint, weights_only=True)
        for key, value in [("model", "M3_SoftJointCBM"), ("protocol", "soft_joint_state_weighted_v1"),
                           ("bottleneck", "soft"), ("softmax_temperature", .5)]:
            changed = copy.deepcopy(original)
            changed["config"][key] = value
            torch.save(changed, checkpoint)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "M4 hard/ST protocol"):
                run_m4.evaluate_m4_test(str(checkpoint), str(self.manifest), save_results=False)
        torch.save(original, checkpoint)
        mapping = json.loads((self.path / "label_mapping.json").read_text())
        mapping["pigment_network"]["typical"], mapping["pigment_network"]["atypical"] = (
            mapping["pigment_network"]["atypical"], mapping["pigment_network"]["typical"])
        (self.path / "label_mapping.json").write_text(json.dumps(mapping))
        with self.assertRaisesRegex(ValueError, "schema differs"):
            run_m4.evaluate_m4_test(str(checkpoint), str(self.manifest), save_results=False)

    def test_frozen_export_intervention_and_tamper_rejection(self):
        saved = self.train()
        checkpoint, predictions = self.path / "seed42.pth", self.path / "seed42.json"
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_m4.run_m4_intervention(str(checkpoint), str(predictions), str(self.manifest),
                                               str(self.path / "intervention.json"))
        self.assertEqual(result["num_subsets"], 128)
        self.assertEqual(result["baseline_y_pred"], [r["y_pred"] for r in saved["test_predictions"]])
        np.testing.assert_allclose(result["baseline_y_prob"], [r["y_prob"] for r in saved["test_predictions"]], atol=1e-6)
        full = result["subsets"][-1]
        state = torch.load(checkpoint, weights_only=True)["model_state_dict"]
        head = nn.Linear(28, 2)
        head.load_state_dict({"weight": state["diagnosis_head.weight"], "bias": state["diagnosis_head.bias"]})
        true_vectors = np.concatenate([np.eye(CONCEPT_NUM_CLASSES[c])[
            [r["concept_true"][c] for r in saved["test_predictions"]]] for c in CONCEPT_NAMES], axis=1)
        with torch.no_grad():
            expected = head(torch.tensor(true_vectors, dtype=torch.float32)).softmax(-1)[:, 1].numpy()
        np.testing.assert_allclose(full["y_prob"], expected, atol=1e-7)
        for mutation in ("hash", "soft_vector", "probabilities", "diagnosis", "case_ids"):
            changed = copy.deepcopy(saved)
            row = changed["test_predictions"][0]
            if mutation == "hash": changed["checkpoint_sha256"] = "wrong"
            elif mutation == "soft_vector": row["hard_concept_vector"][:3] = [1/3]*3
            elif mutation == "probabilities": row["concept_probabilities"][CONCEPT_NAMES[0]] = [float("nan")]*3
            elif mutation == "diagnosis": row["y_prob"] = -1.
            else: changed["test_predictions"][-1]["case_num"] = row["case_num"]
            predictions.write_text(json.dumps(changed))
            output = self.path / f"bad_{mutation}.json"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                run_m4.run_m4_intervention(str(checkpoint), str(predictions), str(self.manifest), str(output))
            self.assertFalse(output.exists())
        predictions.write_text(json.dumps(saved))
        wrong_seed_output = self.path / "wrong_seed.json"
        with self.assertRaisesRegex(ValueError, "Requested seed differs"):
            run_m4.run_m4_intervention(str(checkpoint), str(predictions), str(self.manifest),
                                       str(wrong_seed_output), expected_seed=123)
        self.assertFalse(wrong_seed_output.exists())
        with self.assertRaisesRegex(ValueError, "overwrite its inputs"):
            run_m4.run_m4_intervention(str(checkpoint), str(predictions), str(self.manifest), str(predictions), True)

    def test_intervention_cli_needs_no_backbone_or_images(self):
        self.train()
        output = self.path / "cli_intervention.json"
        completed = subprocess.run([sys.executable, "-B", str(ROOT / "experiments/run_m4_st.py"),
            "--mode", "intervention", "--manifest_path", str(self.manifest),
            "--checkpoint_path", str(self.path / "seed42.pth"),
            "--predictions_path", str(self.path / "seed42.json"), "--results_path", str(output)],
            capture_output=True, text=True, timeout=45,
            env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(output.read_text())["num_subsets"], 128)

    def test_hard_intervention_rejects_soft_inputs_and_preserves_unedited_groups(self):
        class RecordingHead(nn.Linear):
            def __init__(self):
                super().__init__(28, 2)
                self.inputs = []
            def forward(self, x):
                self.inputs.append(x.clone())
                return super().forward(x)
        head = RecordingHead()
        vectors = np.concatenate([np.eye(k)[[0, 1]] for k in CONCEPT_NUM_CLASSES.values()], axis=1)
        targets = np.zeros((2, 7), dtype=int)
        result = evaluate_hard_interventions(head, vectors, targets, [0, 1], self.schema, .8, [10, 20])
        self.assertEqual(len(result["subsets"]), 128)
        for row, actual in zip(result["subsets"], head.inputs[1:]):
            for c in CONCEPT_NAMES:
                start, k = self.schema["offsets"][c], CONCEPT_NUM_CLASSES[c]
                expected = np.eye(k)[[0, 0]] if c in row["groups"] else vectors[:, start:start+k]
                np.testing.assert_array_equal(actual[:, start:start+k].numpy(), expected)
        soft = vectors.copy()
        soft[:, :3] = 1/3
        with self.assertRaisesRegex(ValueError, "one-hot"):
            evaluate_hard_interventions(head, soft, targets, [0, 1], self.schema, .8, [10, 20])

    def test_noop_interventions_preserve_frozen_scores_at_rounding_boundary(self):
        head = nn.Linear(28, 2)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
        vectors = np.concatenate([np.eye(k)[[0, 1]] for k in CONCEPT_NUM_CLASSES.values()], axis=1)
        targets = np.array([[0]*7, [1]*7])
        frozen = np.array([np.nextafter(np.float32(.5), np.float32(1)),
                           np.nextafter(np.float32(.5), np.float32(0))], dtype=float)
        result = evaluate_hard_interventions(head, vectors, targets, [1, 0], self.schema, .5, [10, 20],
                                            baseline_probabilities=frozen)
        self.assertGreater(result["baseline_numerics"]["head_replay_max_abs_error"], 0)
        self.assertEqual(result["baseline_y_pred"], [1, 0])
        for row in result["subsets"]:
            np.testing.assert_array_equal(row["y_prob"], frozen)
            self.assertEqual(row["y_pred"], [1, 0])
            self.assertEqual(row["probability_delta"], [0., 0.])
            self.assertEqual(row["worsened_case_ids"], [])
        with self.assertRaisesRegex(ValueError, "numerical tolerance"):
            evaluate_hard_interventions(head, vectors, targets, [1, 0], self.schema, .5, [10, 20],
                                        baseline_probabilities=[.6, .4])

    def test_three_seed_summaries_and_m4_report(self):
        paths = []
        for seed in (42, 123, 2026):
            self.train(seed)
            paths.append(self.path / f"seed{seed}.json")
        with contextlib.redirect_stdout(io.StringIO()):
            diagnosis = summarize_seeds(paths, self.manifest, self.path / "diagnosis_summary.json")
            result = run_m4_interventions.main(["--manifest_path", str(self.manifest),
                "--checkpoints", *[str(self.path / f"seed{s}.pth") for s in (42, 123, 2026)],
                "--predictions", *map(str, paths), "--output_dir", str(self.path / "report")])
        self.assertEqual(diagnosis["seeds"], [42, 123, 2026])
        self.assertEqual(result["model"], run_m4.M4_MODEL)
        self.assertEqual(result["num_cases"], 395)
        report = (self.path / "report/report.md").read_text()
        self.assertIn("Diagnosis head M4-ST", report)
        self.assertNotIn("Diagnosis head M3", report)
        self.assertIn("predicted one-hot", report)
        for file in ("summary.json", "curve.csv", "per_concept.csv", "full_intervention_cases.csv",
                     "intervention_curve.png", "intervention_curve.svg"):
            self.assertGreater((self.path / "report" / file).stat().st_size, 0)
        # A summary must not accept plausible diagnosis metrics while averaging
        # a fabricated concept F1 or a soft vector masquerading as M4 input.
        original = json.loads(paths[-1].read_text())
        for mutation in ("concept_f1", "soft_vector"):
            changed = copy.deepcopy(original)
            if mutation == "concept_f1":
                changed["test_concept_metrics"]["overall_f1_macro_all_defined"] = .987654
            else:
                changed["test_predictions"][0]["hard_concept_vector"][:3] = [1/3]*3
            paths[-1].write_text(json.dumps(changed))
            output = self.path / f"bad_summary_{mutation}.json"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summarize_seeds(paths, self.manifest, output)
            self.assertFalse(output.exists())
        paths[-1].write_text(json.dumps(original))

    def test_analytic_hard_intervention_curve_and_m3_report_regression(self):
        vectors = np.concatenate([np.eye(k)[[0]*4] for k in CONCEPT_NUM_CLASSES.values()], axis=1)
        targets = np.zeros((4, 7), dtype=int)
        targets[:, 0] = [0, 1, 0, 1]
        head = nn.Linear(28, 2)
        with torch.no_grad():
            head.weight.zero_(); head.bias.zero_()
            head.weight[1, 0], head.weight[1, 1] = -2., 2.
        runs = []
        source = self.path / "source.json"
        source.write_text(json.dumps({"test_predictions": [{"case_num": i, "is_inconsistent": False} for i in range(4)]}))
        for seed in (42, 123, 2026):
            run = evaluate_hard_interventions(head, vectors, targets, [0, 1, 0, 1], self.schema, .5, list(range(4)))
            run.update(model=run_m4.M4_MODEL, mode="intervention_analysis",
                hyperparameters={"seed": seed, "protocol": "fixture"}, concept_schema=self.schema,
                manifest_fingerprint=manifest_fingerprint(self.manifest), checkpoint_sha256=str(seed),
                predictions_path=str(source))
            runs.append(run)
        summary = summarize_interventions(runs)
        for row in summary["curve"]:
            self.assertAlmostEqual(row["balanced_accuracy"]["mean"], .5+.5*row["m"]/7)
        self.assertAlmostEqual(summary["intervention_curve_auc"]["balanced_accuracy"]["absolute"]["mean"], .75)
        # The shared exporter must retain the original M3 interpretation by default.
        soft_runs = copy.deepcopy(runs)
        for run in soft_runs: run["model"] = "M3_SoftJointCBM"
        soft_summary = summarize_interventions(soft_runs)
        output = self.path / "m3_report"
        output.mkdir()
        export_report(soft_summary, soft_runs, output)
        report = (output / "report.md").read_text()
        self.assertIn("Diagnosis head M3 được học từ soft probabilities", report)
        self.assertNotIn("Diagnosis head M4", report)


if __name__ == "__main__":
    unittest.main()
