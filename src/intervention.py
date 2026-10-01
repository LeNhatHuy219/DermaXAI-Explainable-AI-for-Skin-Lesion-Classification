"""Exhaustive categorical group intervention for a frozen soft CBM head."""
from itertools import combinations
import numpy as np
import torch

from .metrics import compute_metrics


@torch.no_grad()
def evaluate_soft_interventions(head, probabilities, true_indices, labels, schema,
                                 threshold, case_ids):
    names = schema["concept_names"]
    count = len(names)
    device = next(head.parameters()).device
    head.eval()
    vectors = torch.as_tensor(probabilities, dtype=torch.float32, device=device)
    true_indices = np.asarray(true_indices, dtype=int)
    labels = np.asarray(labels, dtype=int)
    if vectors.shape != (len(labels), schema["total_states"]) or true_indices.shape != (len(labels), count):
        raise ValueError("Misaligned concept vectors, ground-truth indices or labels")
    if len(case_ids) != len(labels) or len(set(case_ids)) != len(case_ids) or len(labels) == 0:
        raise ValueError("Expected unique aligned nonempty case IDs")
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Expected binary labels and a finite threshold in [0,1]")
    original_pred = []
    for i, name in enumerate(names):
        start = schema["offsets"][name]
        k = schema["concept_num_classes"][name]
        block = vectors[:, start:start+k]
        if (not torch.isfinite(block).all() or torch.any(block < 0) or torch.any(block > 1) or
                not torch.allclose(block.sum(-1), torch.ones(len(labels), device=device), atol=1e-5)):
            raise ValueError(f"Invalid categorical probability block: {name}")
        if np.any((true_indices[:, i] < 0) | (true_indices[:, i] >= k)):
            raise ValueError(f"Invalid ground-truth state index: {name}")
        original_pred.append(block.argmax(-1).cpu().numpy())
    original_correct = np.stack(original_pred, axis=1) == true_indices
    baseline_probs = head(vectors).softmax(-1)[:, 1].cpu().numpy()
    baseline_preds = (baseline_probs >= threshold).astype(int)
    baseline_correct = baseline_preds == labels
    rows = []
    for m in range(count + 1):
        for subset in combinations(range(count), m):
            corrected = vectors.clone()
            for i in subset:
                name = names[i]
                k = schema["concept_num_classes"][name]
                start = schema["offsets"][name]
                targets = torch.as_tensor(true_indices[:, i], dtype=torch.long, device=device)
                corrected[:, start:start+k] = torch.nn.functional.one_hot(targets, k).to(corrected.dtype)
            probs = head(corrected).softmax(-1)[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)
            correct = preds == labels
            remaining = [i for i in range(count) if i not in subset]
            rows.append({"groups": [names[i] for i in subset], "m": m, "rate": m/count,
                         "diagnosis_metrics": compute_metrics(labels, preds, probs),
                         "remaining_concept_accuracy": (float(original_correct[:, remaining].mean())
                                                        if remaining else None),
                         "improved_case_ids": [int(case_ids[i]) for i in np.flatnonzero(~baseline_correct & correct)],
                         "worsened_case_ids": [int(case_ids[i]) for i in np.flatnonzero(baseline_correct & ~correct)],
                         "y_pred": preds.tolist(), "y_prob": probs.astype(float).tolist(),
                         "probability_delta": (probs-baseline_probs).astype(float).tolist()})
    metric_names = ["accuracy", "balanced_accuracy", "f1_macro", "sensitivity", "specificity", "roc_auc", "pr_auc"]
    curve = []
    for m in range(count + 1):
        subsets = [row for row in rows if row["m"] == m]
        entry = {"m": m, "rate": m/count, "num_subsets": len(subsets), "diagnosis_metrics": {}}
        for key in metric_names:
            values = [row["diagnosis_metrics"][key] for row in subsets]
            entry["diagnosis_metrics"][key] = {"mean": float(np.mean(values)), "sd_across_subsets": float(np.std(values, ddof=0))}
        values = [row["remaining_concept_accuracy"] for row in subsets]
        entry["remaining_concept_accuracy"] = ({"mean": float(np.mean(values)),
                                                "sd_across_subsets": float(np.std(values, ddof=0))}
                                               if m < count else None)
        curve.append(entry)
    auc = {}
    rates = [row["rate"] for row in curve]
    for key in ["balanced_accuracy", "f1_macro"]:
        values = [row["diagnosis_metrics"][key]["mean"] for row in curve]
        # NumPy 2.4 removed trapz; retain compatibility with NumPy 1.24.
        integrate = getattr(np, "trapezoid", None)
        if integrate is None:
            integrate = np.trapz
        area = float(integrate(values, x=rates))
        auc[key] = {"absolute": area, "gain_over_baseline": area-values[0]}
    return {"case_ids": [int(x) for x in case_ids], "y_true": labels.tolist(),
            "baseline_y_pred": baseline_preds.tolist(), "baseline_y_prob": baseline_probs.astype(float).tolist(),
            "decision_threshold": float(threshold), "num_subsets": len(rows),
            "subset_sd_definition": "population SD across exhaustive subsets; not seed SD or sample CI",
            "subsets": rows, "curve": curve, "intervention_curve_auc": auc}
