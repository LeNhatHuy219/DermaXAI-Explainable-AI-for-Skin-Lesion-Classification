# Kiểm tra protocol M4-SG: chặn diagnosis gradient và train/test/intervention.
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

from src.cbm import HardJointCBM, HardStopGradientCBM, make_diagnosis_head
from src.protocol import load_concept_schema
from experiments import (m4_common, run_m4_st, run_m4_sg as run_m4,
                         run_m4_sg_interventions as run_m4_interventions)
from experiments.summarize_seeds import summarize_seeds

ROOT = Path(__file__).resolve().parents[1]


class TinyHardStopGradientCBM(HardStopGradientCBM):
    """Use the production SG forward with a small trainable feature extractor."""
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
        frame = pd.read_csv(manifest_path)
        self.df = frame[frame.split == split].reset_index(drop=True)
        self.label_mapping = load_concept_schema(manifest_path)["label_mapping"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return {"image": torch.tensor([index/1000, 1., -1.]),
                "label": torch.tensor(int(row.diagnosis_binary)),
                "concept_indices": torch.tensor([self.label_mapping[c][row[c]] for c in CONCEPT_NAMES])}


class M4SGProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m4_sg_protocol_")
        self.path = Path(self.temp.name)
        self.manifest = self.path / "manifest.csv"
        shutil.copy2(ROOT / "data/manifest.csv", self.manifest)
        shutil.copy2(ROOT / "data/label_mapping.json", self.path / "label_mapping.json")
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
             patch.object(m4_common, "get_hard_stop_gradient_cbm", TinyHardStopGradientCBM), \
             patch.object(m4_common, "get_hard_joint_cbm", side_effect=AssertionError("SG must use the SG model")), \
             patch.object(m4_common, "Derm7ptDataset", TinyDataset), \
             contextlib.redirect_stdout(io.StringIO()):
            yield loader

    def args(self, seed=42):
        return ["--manifest_path", str(self.manifest),
                "--epochs", "1", "--batch_size", "128", "--device", "cpu", "--num_workers", "0",
                "--seed", str(seed), "--checkpoint_path", str(self.path / f"sg_seed{seed}.pth"),
                "--results_path", str(self.path / f"sg_seed{seed}.json")]

    def train(self, seed=42, extra=None):
        with self.pipeline() as loader:
            result = run_m4.main(self.args(seed) + (extra or []))
            self.assertFalse(loader.call_args.kwargs["include_test"])
            return result

    def test_diagnosis_only_step_preserves_backbone_and_concept_parameters(self):
        torch.manual_seed(7)
        model = TinyHardStopGradientCBM().train()
        inputs = torch.tensor([[.1, 1., -1.], [.3, -.4, .2], [-.7, .1, 1.]])
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
        diagnosis, concepts, vector = model(inputs)
        self.assertFalse(vector.requires_grad)
        self.assertTrue(all(logits.requires_grad for logits in concepts.values()))
        nn.functional.cross_entropy(diagnosis, torch.tensor([0, 1, 0])).backward()
        for name, parameter in model.named_parameters():
            if name.startswith("diagnosis_head."):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0.)
            else:
                self.assertIsNone(parameter.grad, name)
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name.startswith("diagnosis_head."):
                self.assertFalse(torch.equal(parameter.detach(), before[name]), name)
            else:
                self.assertTrue(torch.equal(parameter.detach(), before[name]), name)

    def test_separate_head_lr_changes_head_updates_without_changing_sg_concept_updates(self):
        # Cùng batch/khởi tạo/concept loss; chỉ đổi LR diagnosis head.
        # Backbone SG phải cập nhật giống nhau, mọi parameter thuộc đúng một group.
        torch.manual_seed(7)
        models = [TinyHardStopGradientCBM().train()]
        models.append(copy.deepcopy(models[0]))
        optimizers = [m4_common._build_optimizer(model, 1e-4, .01, head_lr)
                      for model, head_lr in zip(models, [1e-4, 1e-3])]
        inputs = torch.tensor([[.1, 1., -1.], [.3, -.4, .2], [-.7, .1, 1.]])
        for model, optimizer in zip(models, optimizers):
            grouped = [p for group in optimizer.param_groups for p in group["params"]]
            self.assertEqual(len(grouped), len({id(p) for p in grouped}))
            self.assertEqual({id(p) for p in grouped}, {id(p) for p in model.parameters()})
            self.assertEqual({id(p) for p in optimizer.param_groups[1]["params"]},
                             {id(p) for p in model.diagnosis_head.parameters()})
            torch.manual_seed(123)
            diagnosis, concepts, _ = model(inputs)
            loss = nn.functional.cross_entropy(diagnosis, torch.tensor([0, 1, 0]))
            loss += torch.stack([nn.functional.cross_entropy(
                concepts[c], torch.tensor([0, 1, 0])) for c in CONCEPT_NAMES]).mean()
            loss.backward()
            optimizer.step()
        for name, parameter in models[0].named_parameters():
            other = dict(models[1].named_parameters())[name]
            if name.startswith("diagnosis_head."):
                self.assertFalse(torch.equal(parameter, other), name)
            else:
                torch.testing.assert_close(parameter, other, atol=0, rtol=0)
        self.assertEqual([group["lr"] for group in optimizers[1].param_groups], [1e-4, 1e-3])

    def test_separate_lr_pilot_freezes_config_and_test_preserves_validation(self):
        # Kiểm tra end-to-end: CLI -> optimizer groups -> checkpoint -> frozen test.
        result = self.train(extra=["--skip_test", "--diagnosis_lr", "0.001", "--epochs", "2"])
        config = result["hyperparameters"]
        self.assertEqual(config["learning_rate"], 1e-4)
        self.assertEqual(config["diagnosis_learning_rate"], 1e-3)
        self.assertEqual(result["history"][0]["learning_rates"], [1e-4, 1e-3])
        self.assertLess(result["history"][1]["learning_rates"][1], 1e-3)
        self.assertIn("validation_f1_macro_at_0.5", result["history"][0])
        checkpoint = torch.load(self.path / "sg_seed42.pth", weights_only=True)
        self.assertEqual(len(checkpoint["optimizer_state_dict"]["param_groups"]), 2)
        with self.pipeline():
            run_m4.main(self.args() + ["--mode", "test", "--overwrite"])
        saved = json.loads((self.path / "sg_seed42.json").read_text())
        self.assertEqual(saved["validation_predictions"], result["validation_predictions"])
        self.assertEqual(saved["hyperparameters"]["diagnosis_learning_rate"], 1e-3)
        run_m4.validate_m4_sg_test_export(saved, self.manifest)

    def test_mlp_sg_head_preserves_gradient_isolation(self):
        # Diagnosis MLP có gradient; backbone/concept vẫn chỉ học từ concept loss.
        torch.manual_seed(7)
        model = TinyHardStopGradientCBM(diagnosis_head="mlp128").train()
        diagnosis, concepts, vector = model(torch.randn(3, 3))
        self.assertFalse(vector.requires_grad)
        nn.functional.cross_entropy(diagnosis, torch.tensor([0, 1, 0])).backward()
        self.assertTrue(all(p.grad is None for p in model.features.parameters()))
        self.assertTrue(all(p.grad is None for p in model.concept_heads.parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.diagnosis_head.parameters()), 0.)

    def test_mlp_sg_train_test_intervention_and_invalid_specification(self):
        result = self.train(extra=["--diagnosis_head", "mlp128", "--diagnosis_lr", "0.001"])
        self.assertEqual(result["hyperparameters"]["diagnosis_head"], "MLP(28, 128, 2)")
        output = self.path / "mlp_sg_intervention.json"
        run = run_m4.run_m4_sg_intervention(str(self.path / "sg_seed42.pth"),
            str(self.path / "sg_seed42.json"), str(self.manifest), str(output))
        self.assertEqual(run["num_subsets"], 128)
        self.assertEqual(run["baseline_y_pred"], [row["y_pred"] for row in result["test_predictions"]])
        original = torch.load(self.path / "sg_seed42.pth", weights_only=True)
        original["config"]["diagnosis_dropout"] = .2
        torch.save(original, self.path / "sg_seed42.pth")
        with patch.object(m4_common, "Derm7ptDataset", side_effect=AssertionError("No test images")):
            with self.assertRaisesRegex(ValueError, "MLP128"):
                run_m4.evaluate_m4_sg_test(str(self.path / "sg_seed42.pth"), str(self.manifest), save_results=False)

    def test_head_lr_paths_are_distinct_and_invalid_lr_fails_before_loading_data(self):
        # Runs mới tách khỏi baseline, khác LR/epochs phải có đường dẫn khác nhau.
        base = m4_common._default_paths(gradient_mode="stop_gradient")
        candidates = [m4_common._default_paths(gradient_mode="stop_gradient",
                      diagnosis_lr=lr, epochs=epochs)
                      for lr, epochs in [(1e-4, 20), (5e-4, 20), (1e-3, 20), (1e-3, 10)]]
        self.assertEqual(len({path for pair in [base, *candidates] for path in pair}), 10)
        with patch.object(m4_common, "get_dataloaders", side_effect=AssertionError("No training")):
            for value in [0., -1., float("nan"), float("inf")]:
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "diagnosis learning rate"):
                    m4_common.run_m4_experiment(diagnosis_lr=value, gradient_mode="stop_gradient")

    def test_joint_loss_backbone_gradient_is_exactly_the_weighted_concept_gradient(self):
        torch.manual_seed(7)
        model = TinyHardStopGradientCBM().train()
        diagnosis, concepts, _ = model(torch.randn(3, 3))
        concept_loss = torch.stack([nn.functional.cross_entropy(
            concepts[c], torch.tensor([0, 1, 0])) for c in CONCEPT_NAMES]).mean()
        diagnosis_loss = nn.functional.cross_entropy(diagnosis, torch.tensor([0, 1, 0]))
        upstream = [model.features.weight, *[model.concept_heads[c].weight for c in CONCEPT_NAMES]]
        concept_gradients = torch.autograd.grad(concept_loss, upstream, retain_graph=True)
        joint_gradients = torch.autograd.grad(diagnosis_loss + .5*concept_loss, upstream, retain_graph=True)
        for expected, actual in zip(concept_gradients, joint_gradients):
            self.assertGreater(actual.abs().sum().item(), 0.)
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, .5*expected, atol=1e-7, rtol=1e-6)
        (diagnosis_loss + .5*concept_loss).backward()
        self.assertGreater(model.diagnosis_head.weight.grad.abs().sum().item(), 0.)

    def test_real_backbone_initialization_forward_and_gradient_isolation(self):
        torch.manual_seed(42)
        st = HardJointCBM(CONCEPT_NAMES, CONCEPT_NUM_CLASSES, pretrained=False).train()
        torch.manual_seed(42)
        sg = HardStopGradientCBM(CONCEPT_NAMES, CONCEPT_NUM_CLASSES, pretrained=False).train()
        for name, value in st.state_dict().items():
            self.assertTrue(torch.equal(value, sg.state_dict()[name]), name)
        images = torch.randn(2, 3, 64, 64)
        torch.manual_seed(123)
        st_diagnosis, st_concepts, st_vector = st(images)
        torch.manual_seed(123)
        sg_diagnosis, sg_concepts, sg_vector = sg(images)
        self.assertTrue(torch.equal(st_diagnosis, sg_diagnosis))
        self.assertTrue(torch.equal(st_vector, sg_vector))
        self.assertTrue(torch.all((sg_vector == 0) | (sg_vector == 1)))
        self.assertTrue(torch.equal(sg_vector.sum(-1), torch.full((2,), 7.)))
        self.assertTrue(st_vector.requires_grad)
        self.assertFalse(sg_vector.requires_grad)
        for name in CONCEPT_NAMES:
            self.assertTrue(torch.equal(st_concepts[name], sg_concepts[name]))
        nn.functional.cross_entropy(sg_diagnosis, torch.tensor([0, 1])).backward()
        self.assertTrue(all(p.grad is None for p in sg.features.parameters()))
        self.assertTrue(all(p.grad is None for p in sg.concept_heads.parameters()))
        self.assertGreater(sg.diagnosis_head.weight.grad.abs().sum().item(), 0.)
        sg.zero_grad(set_to_none=True)
        _, logits, _ = sg(images)
        torch.stack([nn.functional.cross_entropy(logits[c], torch.tensor([0, 1]))
                     for c in CONCEPT_NAMES]).mean().backward()
        backbone_gradient = next(sg.features.parameters()).grad
        self.assertIsNotNone(backbone_gradient)
        self.assertTrue(torch.isfinite(backbone_gradient).all())
        self.assertGreater(backbone_gradient.abs().sum().item(), 0.)
        for c in CONCEPT_NAMES:
            gradient = sg.concept_heads[c].weight.grad
            self.assertGreater(gradient.abs().sum().item(), 0.)
        self.assertIsNone(sg.diagnosis_head.weight.grad)

    def test_variant_paths_and_invalid_mode_before_training(self):
        st_paths = run_m4_st._default_paths(42)
        self.assertIn("m4_hard_joint_cbm_seed42", st_paths[0])
        sg_paths = run_m4._default_paths(42)
        self.assertIn("m4_hard_sg_cbm_seed42", sg_paths[0])
        self.assertTrue(set(st_paths).isdisjoint(sg_paths))
        self.assertNotEqual(sg_paths, run_m4._default_paths(123))
        self.assertNotEqual(sg_paths, run_m4._default_paths(42, "comparison"))
        with patch.object(m4_common, "get_dataloaders", side_effect=AssertionError("No training")):
            with self.assertRaisesRegex(ValueError, "gradient mode"):
                m4_common.run_m4_experiment(gradient_mode="unknown")

    def test_pilot_then_frozen_test_uses_sg_and_preserves_validation(self):
        result = self.train(extra=["--skip_test"])
        self.assertEqual(result["model"], run_m4.M4_MODEL)
        self.assertEqual(result["hyperparameters"]["protocol"], run_m4.M4_PROTOCOL)
        self.assertEqual(result["hyperparameters"]["gradient_estimator"], "stop_gradient")
        self.assertEqual(result["hyperparameters"]["diagnosis_head"], "Linear(28, 2)")
        self.assertNotIn("test_predictions", result)
        original = json.loads((self.path / "sg_seed42.json").read_text())
        with self.pipeline():
            combined = run_m4.main(self.args() + ["--mode", "test", "--overwrite"])
        self.assertEqual(combined["model"], run_m4.M4_MODEL)
        self.assertEqual(combined["mode"], "final_test")
        saved = json.loads((self.path / "sg_seed42.json").read_text())
        self.assertEqual(saved["mode"], "train_validation_test")
        self.assertEqual(saved["validation_predictions"], original["validation_predictions"])
        self.assertEqual(saved["decision_threshold"], original["decision_threshold"])
        self.assertEqual(len(saved["test_predictions"]), 395)
        run_m4.validate_m4_sg_test_export(saved, self.manifest)

    def test_inconsistent_variant_identity_and_cross_variant_exports_are_rejected(self):
        result = self.train()
        checkpoint = self.path / "sg_seed42.pth"
        original = torch.load(checkpoint, weights_only=True)
        for key, value in [("model", run_m4_st.M4_MODEL), ("protocol", run_m4_st.M4_PROTOCOL),
                           ("gradient_estimator", "deterministic_softmax_straight_through")]:
            changed = copy.deepcopy(original)
            changed["config"][key] = value
            torch.save(changed, checkpoint)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "protocol"):
                run_m4.evaluate_m4_sg_test(str(checkpoint), str(self.manifest), save_results=False)
        torch.save(original, checkpoint)
        wrong_export = copy.deepcopy(result)
        wrong_export["model"] = run_m4_st.M4_MODEL
        wrong_export["hyperparameters"].update(m4_common.M4_VARIANTS["straight_through"])
        predictions = self.path / "wrong_st_export.json"
        predictions.write_text(json.dumps(wrong_export))
        output = self.path / "wrong_pair.json"
        with self.assertRaisesRegex(ValueError, "frozen M4"):
            run_m4.run_m4_sg_intervention(str(checkpoint), str(predictions), str(self.manifest), str(output))
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, "gradient mode differs"):
            run_m4_st.run_m4_intervention(str(checkpoint), str(self.path / "sg_seed42.json"),
                                        str(self.manifest), str(output))
        self.assertFalse(output.exists())

    def test_test_mode_routes_custom_sg_checkpoint_to_sg_default_results(self):
        self.train(extra=["--skip_test"])
        # The dedicated SG runner routes its fixed variant to SG outputs.
        with self.pipeline(), patch.object(m4_common, "ROOT", self.path):
            result = run_m4.main(["--mode", "test", "--manifest_path", str(self.manifest),
                "--checkpoint_path", str(self.path / "sg_seed42.pth"), "--device", "cpu"])
        sg_result = self.path / "results/f1_macro/m4_sg/linear_e1/seed42.json"
        st_result = self.path / "results/f1_macro/m4_st/linear_e1/seed42.json"
        self.assertTrue(sg_result.exists())
        self.assertFalse(st_result.exists())
        self.assertEqual(result["model"], run_m4.M4_MODEL)

    def test_dedicated_runners_reject_variant_switches_and_opposite_checkpoints(self):
        for runner in (run_m4_st, run_m4):
            with self.subTest(runner=runner.__name__), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runner.main(["--gradient_mode", "stop_gradient"])
                self.assertEqual(error.exception.code, 2)
        self.train(extra=["--skip_test"])
        sg_checkpoint = self.path / "sg_seed42.pth"
        st_fixture = torch.load(sg_checkpoint, weights_only=True)
        st_fixture["config"].update(m4_common.M4_VARIANTS["straight_through"])
        st_checkpoint = self.path / "st_fixture.pth"
        torch.save(st_fixture, st_checkpoint)
        with patch.object(m4_common, "Derm7ptDataset", side_effect=AssertionError("No test images")):
            for runner, checkpoint in ((run_m4_st, sg_checkpoint), (run_m4, st_checkpoint)):
                output = self.path / f"wrong_runner_{runner.__name__}.json"
                with self.subTest(runner=runner.__name__), self.assertRaisesRegex(ValueError, "gradient mode differs"):
                    runner.main(["--mode", "test", "--device", "cpu", "--manifest_path", str(self.manifest),
                                 "--checkpoint_path", str(checkpoint), "--results_path", str(output)])
                self.assertFalse(output.exists())

    def test_standalone_sg_intervention_needs_no_backbone_or_images(self):
        self.train()
        output = self.path / "sg_cli_intervention.json"
        completed = subprocess.run([sys.executable, "-B", str(ROOT / "experiments/run_m4_sg.py"),
            "--mode", "intervention", "--manifest_path", str(self.manifest),
            "--checkpoint_path", str(self.path / "sg_seed42.pth"),
            "--predictions_path", str(self.path / "sg_seed42.json"), "--results_path", str(output)],
            capture_output=True, text=True, timeout=45,
            env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(output.read_text())
        self.assertEqual(result["model"], run_m4.M4_MODEL)
        self.assertEqual(result["num_subsets"], 128)

    def test_sg_summary_interventions_and_tamper_rejection(self):
        seeds = (42, 123, 2026)
        paths = []
        for seed in seeds:
            self.train(seed)
            paths.append(self.path / f"sg_seed{seed}.json")
        with contextlib.redirect_stdout(io.StringIO()):
            diagnosis = summarize_seeds(paths, self.manifest, self.path / "sg_summary.json")
            intervention = run_m4_interventions.main([
                "--manifest_path", str(self.manifest),
                "--checkpoints", *[str(self.path / f"sg_seed{s}.pth") for s in seeds],
                "--predictions", *map(str, paths), "--output_dir", str(self.path / "sg_report")])
        self.assertEqual(diagnosis["model"], run_m4.M4_MODEL)
        self.assertEqual(diagnosis["seeds"], list(seeds))
        self.assertIn("concept_metrics", diagnosis)
        self.assertEqual(intervention["model"], run_m4.M4_MODEL)
        for seed, path in zip(seeds, paths):
            run = json.loads((self.path / f"sg_report/seed{seed}.json").read_text())
            source = json.loads(path.read_text())
            self.assertEqual(run["num_subsets"], 128)
            self.assertEqual(run["baseline_y_pred"], [r["y_pred"] for r in source["test_predictions"]])
            np.testing.assert_allclose(run["baseline_y_prob"],
                                       [r["y_prob"] for r in source["test_predictions"]], atol=1e-6)
        report = (self.path / "sg_report/report.md").read_text()
        self.assertIn("Diagnosis head M4-SG", report)
        self.assertIn("gradient của loss chẩn đoán dừng tại bottleneck", report)
        for filename in ("summary.json", "curve.csv", "per_concept.csv", "full_intervention_cases.csv",
                         "intervention_curve.png", "intervention_curve.svg"):
            self.assertGreater((self.path / "sg_report" / filename).stat().st_size, 0)
        original = json.loads(paths[-1].read_text())
        for mutation in ("mixed_model", "concept_f1", "soft_vector"):
            changed = copy.deepcopy(original)
            if mutation == "mixed_model":
                changed["model"] = run_m4_st.M4_MODEL
                changed["hyperparameters"].update(m4_common.M4_VARIANTS["straight_through"])
            elif mutation == "concept_f1":
                changed["test_concept_metrics"]["overall_f1_macro_all_defined"] = .987654
            else:
                changed["test_predictions"][0]["hard_concept_vector"][:3] = [1/3]*3
            paths[-1].write_text(json.dumps(changed))
            output = self.path / f"bad_{mutation}.json"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summarize_seeds(paths, self.manifest, output)
            self.assertFalse(output.exists())
        paths[-1].write_text(json.dumps(original))


if __name__ == "__main__":
    unittest.main()
