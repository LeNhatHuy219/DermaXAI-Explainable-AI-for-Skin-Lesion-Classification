"""M2 oracle MLP: ground-truth concepts; train → validation selection → automatic test."""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import compute_diagnosis_weights
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.metrics import compute_metrics, print_metrics_table
from src.protocol import (load_concept_schema, manifest_fingerprint, prepare_training_outputs,
                          save_experiment_results,
                          validate_manifest, validate_concept_schema)
from experiments.run_m1 import get_device, set_seed
from experiments.run_m2_lr import (build_oracle_features, select_validation_threshold,
                               _extract_sample_predictions, _prepare_output, _save_json)


def default_paths(seed=42):
    stem = f"m2_oracle_mlp_seed{seed}"
    return tuple(str(Path(PROJECT_ROOT) / folder / filename) for folder, filename in [
        ("checkpoints", stem + "_best.pth"), ("results", stem + "_results.json")])


def build_model(config):
    return nn.Sequential(nn.Linear(28, config["hidden_dim"]), nn.ReLU(),
                         nn.Dropout(config["dropout"]), nn.Linear(config["hidden_dim"], 2))


@torch.no_grad()
def predict_probabilities(model, features, device):
    model.eval()
    return model(torch.as_tensor(features, dtype=torch.float32, device=device)).softmax(-1)[:, 1].cpu().numpy()


def run_m2_mlp_experiment(manifest_path=None, seed=42, epochs=100, hidden_dim=32,
                          dropout=0.2, lr=1e-3, weight_decay=1e-2, batch_size=32,
                          device_name="auto", checkpoint_path=None, results_path=None,
                          save_results=True, overwrite=False):
    if epochs < 1 or hidden_dim < 1 or batch_size < 1 or not 0 <= dropout < 1:
        raise ValueError("Invalid epochs, hidden_dim, batch_size or dropout")
    if not np.isfinite(lr) or lr <= 0 or not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Invalid learning rate or weight decay")
    checkpoint_path, results_path = prepare_training_outputs(
        default_paths(seed), checkpoint_path, results_path, overwrite, save_results,
    )
    set_seed(seed)
    device = get_device(device_name)
    manifest_path = os.path.abspath(manifest_path or str(Path(PROJECT_ROOT) / "data/manifest.csv"))
    schema = load_concept_schema(manifest_path)
    df = pd.read_csv(manifest_path, dtype={"is_inconsistent_profile": bool})
    train = df[df.split == "train"].reset_index(drop=True)
    valid = df[df.split == "valid"].reset_index(drop=True)
    if len(train) != 413 or len(valid) != 203:
        raise ValueError("Expected official train/validation split sizes 413/203")
    x_train = build_oracle_features(train, schema["label_mapping"])
    x_valid = build_oracle_features(valid, schema["label_mapping"])
    y_train = torch.as_tensor(train.diagnosis_binary.to_numpy(), dtype=torch.long, device=device)
    y_valid = valid.diagnosis_binary.to_numpy(dtype=int)
    x_tensor = torch.as_tensor(x_train, device=device)
    weights = compute_diagnosis_weights(manifest_path, PROJECT_ROOT).to(device)
    config = {"model_type": "OracleMLP", "seed": seed, "hidden_dim": hidden_dim,
              "dropout": dropout, "epochs": epochs, "batch_size": batch_size,
              "learning_rate": lr, "weight_decay": weight_decay,
              "feature_space": "concept_onehot_28d_ground_truth",
              "class_weights": weights.cpu().tolist(), "optimizer": "AdamW",
              "checkpoint_selection": "validation_balanced_accuracy_at_0.5",
              "threshold_selection": "maximize_validation_balanced_accuracy_tie_nearest_0.5",
              "torch_version": str(torch.__version__), "device": str(device)}
    _prepare_output(checkpoint_path, overwrite)
    if save_results:
        _prepare_output(results_path, overwrite)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best_score = -1.0
    history = []
    fingerprint = manifest_fingerprint(manifest_path)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train), device=device)
        for indices in order.split(batch_size):
            optimizer.zero_grad()
            loss = criterion(model(x_tensor[indices]), y_train[indices])
            loss.backward()
            optimizer.step()
        probs = predict_probabilities(model, x_valid, device)
        metrics = compute_metrics(y_valid, (probs >= 0.5).astype(int), probs)
        score = metrics["balanced_accuracy"]
        history.append({"epoch": epoch, "validation_balanced_accuracy_at_0.5": score})
        if score > best_score:
            best_score = score
            torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch,
                        "concept_schema": schema, "manifest_fingerprint": fingerprint,
                        "manifest_sha256": fingerprint["raw_sha256"]}, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    probs = predict_probabilities(model, x_valid, device)
    threshold, _ = select_validation_threshold(y_valid, probs)
    preds = (probs >= threshold).astype(int)
    checkpoint["decision_threshold"] = threshold
    torch.save(checkpoint, checkpoint_path)
    results = {"model": "M2_Oracle_MLP", "mode": "train_validation", "hyperparameters": config,
               "best_epoch": checkpoint["epoch"], "decision_threshold": threshold,
               "checkpoint_path": os.path.abspath(checkpoint_path), "concept_schema": schema,
               "manifest_sha256": fingerprint["raw_sha256"], "manifest_fingerprint": fingerprint,
               "validation_metrics_at_0.5": compute_metrics(y_valid, (probs >= 0.5).astype(int), probs),
               "validation_metrics": compute_metrics(y_valid, preds, probs),
               "validation_predictions": _extract_sample_predictions(valid, preds, probs), "history": history}
    if save_results:
        _save_json(results_path, results)
    print(f"M2 MLP: best epoch={checkpoint['epoch']}, validation BAcc={results['validation_metrics']['balanced_accuracy']:.4f}")
    print_metrics_table(results["validation_metrics"],
                        title=f"Validation Set (seed={seed}, epoch={checkpoint['epoch']}, threshold={threshold:.6f})")
    print(f"Checkpoint đã lưu tại: {os.path.abspath(checkpoint_path)}")
    if save_results:
        print(f"Kết quả validation đã lưu tại: {os.path.abspath(results_path)}")
    return results


def evaluate_m2_mlp_test(checkpoint_path, manifest_path=None, device_name="auto",
                        results_path=None, save_results=True, overwrite=False):
    device = get_device(device_name)
    manifest_path = os.path.abspath(manifest_path or str(Path(PROJECT_ROOT) / "data/manifest.csv"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    validate_manifest(checkpoint, manifest_path)
    schema = validate_concept_schema(checkpoint, manifest_path)
    threshold = float(checkpoint["decision_threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid frozen diagnosis threshold")
    config = checkpoint["config"]
    results_path = results_path or default_paths(config["seed"])[1]
    if save_results:
        _prepare_output(results_path, overwrite)
    df = pd.read_csv(manifest_path, dtype={"is_inconsistent_profile": bool})
    test = df[df.split == "test"].reset_index(drop=True)
    if len(test) != 395:
        raise ValueError("Expected official test split size 395")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    probs = predict_probabilities(model, build_oracle_features(test, schema["label_mapping"]), device)
    preds = (probs >= threshold).astype(int)
    results = {"model": "M2_Oracle_MLP", "mode": "final_test", "hyperparameters": config,
               "evaluation_device": str(device), "evaluation_torch_version": str(torch.__version__),
               "best_epoch": checkpoint["epoch"], "decision_threshold": threshold,
               "checkpoint_path": os.path.abspath(checkpoint_path), "concept_schema": schema,
               "manifest_sha256": checkpoint["manifest_sha256"],
               "manifest_fingerprint": manifest_fingerprint(manifest_path),
               "test_metrics": compute_metrics(test.diagnosis_binary.to_numpy(dtype=int), preds, probs),
               "test_predictions": _extract_sample_predictions(test, preds, probs)}
    print_metrics_table(results["test_metrics"],
                        title=f"Test Set (seed={config['seed']}, threshold={threshold:.6f}, device={device})")
    if save_results:
        save_experiment_results(results_path, test=results, overwrite=overwrite)
        print(f"Kết quả test đã lưu tại: {os.path.abspath(results_path)}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="train: train rồi tự động test; test: đánh giá checkpoint đã lưu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--manifest_path")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--results_path", help="Một file JSON chứa kết quả validation và test")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    common = dict(manifest_path=args.manifest_path, device_name=args.device, overwrite=args.overwrite)
    if args.mode == "train":
        checkpoint_path, results_path = prepare_training_outputs(
            default_paths(args.seed), args.checkpoint_path, args.results_path,
            args.overwrite,
        )
        validation = run_m2_mlp_experiment(seed=args.seed, epochs=args.epochs, hidden_dim=args.hidden_dim,
                             dropout=args.dropout, batch_size=args.batch_size, lr=args.lr,
                             weight_decay=args.weight_decay, checkpoint_path=checkpoint_path,
                             save_results=False, **common)
        save_experiment_results(results_path, validation=validation, overwrite=args.overwrite)
        test = evaluate_m2_mlp_test(checkpoint_path, save_results=False, **common)
        save_experiment_results(results_path, validation=validation, test=test, overwrite=True)
    else:
        results_path = args.results_path or default_paths(args.seed)[1]
        _prepare_output(results_path, args.overwrite)
        test = evaluate_m2_mlp_test(args.checkpoint_path or default_paths(args.seed)[0],
                                    save_results=False, **common)
        save_experiment_results(results_path, test=test, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
