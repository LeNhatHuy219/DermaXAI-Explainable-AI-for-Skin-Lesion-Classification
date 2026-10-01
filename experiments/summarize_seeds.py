"""Summarize final-test runs with distinct seeds and the same training protocol."""
import argparse
import json
import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.protocol import validate_manifest, validate_concept_schema, manifest_fingerprint
from src.metrics import compute_metrics
import numpy as np
import pandas as pd
from experiments.run_m2_lr import _prepare_output, _save_json


def summarize_seeds(result_paths, manifest_path, output_path, overwrite=False):
    if Path(output_path).resolve() in {Path(path).resolve() for path in result_paths}:
        raise ValueError("Summary must not overwrite a source run")
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in result_paths]
    if len(runs) < 3:
        raise ValueError("At least three runs are required")
    ignore = {"seed", "device", "num_workers", "platform", "torch_version", "torchvision_version",
              "numpy_version", "pandas_version", "scikit_learn_version"}
    model = runs[0]["model"]
    config = {k: v for k, v in runs[0]["hyperparameters"].items() if k not in ignore}
    seeds, labels = set(), None
    schema_hash = runs[0].get("concept_schema", {}).get("sha256")
    test = pd.read_csv(manifest_path)
    test = test[test.split == "test"]
    official_labels = dict(zip(test.case_num.astype(int), test.diagnosis_binary.astype(int)))
    keys = ["accuracy", "balanced_accuracy", "f1_macro", "f1_melanoma", "precision",
            "sensitivity", "specificity", "roc_auc", "pr_auc"]
    for run in runs:
        validate_manifest(run, manifest_path)
        if "concept_schema" in run:
            validate_concept_schema(run, manifest_path)
        if run.get("mode") not in {"final_test", "train_validation_test"} or run["model"] != model:
            raise ValueError("All runs must contain frozen test results of the same model")
        seed = run["hyperparameters"]["seed"]
        if seed in seeds:
            raise ValueError("Duplicate seed")
        seeds.add(seed)
        if {k: v for k, v in run["hyperparameters"].items() if k not in ignore} != config:
            raise ValueError("Training protocols differ; do not pool these runs")
        if run.get("concept_schema", {}).get("sha256") != schema_hash:
            raise ValueError("Concept schemas differ")
        current = {row["case_num"]: row["y_true"] for row in run["test_predictions"]}
        if len(current) != len(run["test_predictions"]) or not current:
            raise ValueError("Duplicate or empty test case IDs")
        if current != official_labels:
            raise ValueError("Export does not match official test cases/labels")
        if labels is not None and labels != current:
            raise ValueError("Test cases/labels differ")
        labels = current
        rows = run["test_predictions"]
        probs = np.asarray([row["y_prob"] for row in rows], dtype=float)
        threshold = float(run["decision_threshold"])
        if not np.isfinite(threshold) or not 0 <= threshold <= 1 or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
            raise ValueError("Invalid probabilities or frozen threshold")
        preds = np.asarray([row["y_pred"] for row in rows])
        if not np.array_equal(preds, (probs >= threshold).astype(int)):
            raise ValueError("Predictions do not use the frozen threshold")
        verified = compute_metrics([row["y_true"] for row in rows], preds, probs)
        if any(not np.isclose(run["test_metrics"][key], verified[key], atol=1e-10) for key in keys):
            raise ValueError("Export metrics do not match per-case predictions")
    metrics = {}
    for key in keys:
        values = [run["test_metrics"][key] for run in runs]
        metrics[key] = {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)), "values": values}
    result = {"model": model, "seeds": [run["hyperparameters"]["seed"] for run in runs],
              "num_seeds": len(runs), "training_protocol": config,
              "manifest_fingerprint": manifest_fingerprint(manifest_path),
              "diagnosis_metrics": metrics, "sd_definition": "sample SD between seeds; not sample CI",
              "source_paths": [str(Path(path).resolve()) for path in result_paths],
              "environments": [{k: run["hyperparameters"].get(k) for k in sorted(ignore) if k != "seed"} for run in runs]}
    if all("test_concept_metrics" in run for run in runs):
        result["concept_metrics"] = {}
        for key in ["overall_concept_accuracy", "overall_f1_macro_all_defined", "exact_match_accuracy"]:
            values = [run["test_concept_metrics"][key] for run in runs]
            result["concept_metrics"][key] = {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)), "values": values}
    _prepare_output(output_path, overwrite)
    _save_json(output_path, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--manifest_path", default=str(Path(PROJECT_ROOT) / "data/manifest.csv"))
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summarize_seeds(args.results, args.manifest_path, args.output_path, args.overwrite)
