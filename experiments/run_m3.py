import argparse
import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Thêm thư mục gốc vào đường dẫn hệ thống
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# QUAN TRỌNG: import src.dataset (kéo theo torch qua src/__init__.py) TRƯỚC sklearn.
# Trên một số máy Windows CPU-only, nạp scikit-learn trước rồi mới nạp torch/torchvision
# trong cùng một process gây xung đột thứ tự nạp DLL OpenMP/MKL dẫn đến segfault. Xem
# ghi chú tương tự tại đầu experiments/run_m2_lr.py.
from src.dataset import (CONCEPT_NAMES, CONCEPT_NUM_CLASSES, Derm7ptDataset,
                         get_dataloaders, compute_concept_statistics)

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from src.metrics import (
    compute_concept_metrics,
    compute_metrics,
    print_concept_metrics_table,
    print_metrics_table,
)
from src.models import SoftJointCBM, get_soft_joint_cbm, load_soft_joint_cbm_state_dict
from src.transforms import get_transforms
from src.protocol import (manifest_fingerprint, load_concept_schema, prepare_training_outputs,
                          save_experiment_results,
                          validate_manifest, validate_concept_schema)
from src.intervention import evaluate_soft_interventions

# M3 - Soft Joint CBM: Ảnh -> 7 concept dự đoán dạng soft probability -> Diagnosis.
# Đây là CBM baseline chính (đối chiếu với M2 Oracle để đánh giá concept sufficiency,
# và với M1 Black-box). Huấn luyện đồng thời
# (joint) toàn bộ backbone + 7 concept head + đầu chẩn đoán g bằng một hàm mất mát tổng
# hợp L = L_diagnosis + lambda * mean(L_concept_i).

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_SEED = 42
DEFAULT_AUGMENTATION_PRESET = "legacy_letterbox"
DEFAULT_CONCEPT_LOSS_WEIGHT = 1.0
M3_PROTOCOL = "soft_joint_state_weighted_v1"


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _manifest_hash(manifest_path: str) -> str:
    digest = hashlib.sha256()
    with open(manifest_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_validation_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Maximize validation balanced accuracy; break ties toward the fixed 0.5 threshold."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) != len(y_prob) or len(np.unique(y_true)) != 2:
        raise ValueError("Threshold selection requires aligned validation scores from both classes")
    if not np.all(np.isfinite(y_prob)) or np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("Predicted probabilities must be finite and in [0, 1]")

    candidates = np.unique(np.concatenate([y_prob, [0.5, 1.0]]))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        score = compute_metrics(y_true, (y_prob >= threshold).astype(int))["balanced_accuracy"]
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12
            and (abs(threshold - 0.5), threshold) < (abs(best_threshold - 0.5), best_threshold)
        ):
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


def compute_train_observed_states(
    train_df: pd.DataFrame, label_mapping: Dict[str, Dict[str, int]]
) -> Dict[str, List[int]]:
    # Trạng thái concept nào thực sự xuất hiện trong train, dùng cho Macro-F1 "train-observed"
    observed: Dict[str, List[int]] = {}
    for c_name in CONCEPT_NAMES:
        observed_vals = train_df[c_name].unique().tolist()
        observed[c_name] = sorted({label_mapping[c_name][v] for v in observed_vals})
    return observed


def _extract_sample_predictions(
    df: pd.DataFrame, y_pred: np.ndarray, y_prob: np.ndarray,
    concept_outputs: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    records = []
    for index, ((_, row), pred, prob) in enumerate(zip(df.iterrows(), y_pred, y_prob)):
        records.append({
            "case_num": int(row["case_num"]),
            "source_index": int(row["source_index"]),
            "split": str(row["split"]),
            "diagnosis": str(row["diagnosis"]),
            "is_inconsistent": bool(row["is_inconsistent_profile"]),
            "y_true": int(row["diagnosis_binary"]),
            "y_pred": int(pred),
            "y_prob": float(prob),
        })
        if concept_outputs is not None:
            records[-1].update({
                "concept_true": {c: int(concept_outputs["true"][c][index]) for c in CONCEPT_NAMES},
                "concept_pred": {c: int(concept_outputs["pred"][c][index]) for c in CONCEPT_NAMES},
                "concept_probabilities": {c: concept_outputs["probabilities"][c][index].tolist()
                                          for c in CONCEPT_NAMES},
            })
    return records


def _concept_loss(
    concept_logits: Dict[str, torch.Tensor],
    concept_indices: torch.Tensor,
    criterion: Dict[str, nn.Module],
) -> torch.Tensor:
    # Multi-Head Cross-Entropy: trung bình cộng CE qua 7 concept head
    losses = [
        criterion[c_name](concept_logits[c_name], concept_indices[:, i])
        for i, c_name in enumerate(CONCEPT_NAMES)
    ]
    return torch.stack(losses).mean()


class StateWeightedCE(nn.Module):
    """Train-state weighted CE; an evaluation batch of zero-weight states yields 0."""
    def __init__(self, weights):
        super().__init__()
        weights = torch.as_tensor(weights, dtype=torch.float32)
        if weights.ndim != 1 or not torch.isfinite(weights).all() or torch.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Expected finite non-negative state weights with positive total")
        self.register_buffer("weights", weights)

    def forward(self, logits, targets):
        loss = nn.functional.cross_entropy(logits, targets, weight=self.weights, reduction="none")
        denominator = self.weights[targets].sum()
        return loss.sum() / denominator.clamp_min(torch.finfo(logits.dtype).eps)


def build_concept_criteria(state_weights, device):
    if not isinstance(state_weights, dict) or set(state_weights) != set(CONCEPT_NAMES):
        raise ValueError("M3 requires train-state weights for all seven concept groups")
    if any(len(state_weights[c]) != CONCEPT_NUM_CLASSES[c] for c in CONCEPT_NAMES):
        raise ValueError("M3 state-weight dimensions differ from concept cardinalities")
    return {c: StateWeightedCE(state_weights[c]).to(device) for c in CONCEPT_NAMES}


def validate_m3_config(config):
    if config.get("protocol") != M3_PROTOCOL or config.get("concept_weighting") != "balanced":
        raise ValueError("Checkpoint is not the finalized weighted M3 protocol; train M3 again")
    return build_concept_criteria(config.get("concept_state_weights"), torch.device("cpu"))


def train_one_epoch(
    model: SoftJointCBM,
    loader: DataLoader,
    diag_criterion: nn.Module,
    concept_criterion: Dict[str, nn.Module],
    concept_loss_weight: float,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    # Huấn luyện 1 epoch, trả về (train_loss, train_diag_loss, train_concept_loss, train_diag_bacc)
    model.train()
    total_loss = 0.0
    total_diag_loss = 0.0
    total_concept_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        concept_indices = batch["concept_indices"].to(device)

        optimizer.zero_grad()
        diag_logits, concept_logits, _ = model(images)
        diag_loss = diag_criterion(diag_logits, labels)
        concept_loss = _concept_loss(concept_logits, concept_indices, concept_criterion)
        loss = diag_loss + concept_loss_weight * concept_loss
        loss.backward()
        optimizer.step()

        bs = len(labels)
        total_loss += loss.item() * bs
        total_diag_loss += diag_loss.item() * bs
        total_concept_loss += concept_loss.item() * bs

        preds = torch.argmax(diag_logits, dim=-1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    n = len(all_labels)
    diag_bacc = compute_metrics(all_labels, all_preds)["balanced_accuracy"]
    return total_loss / n, total_diag_loss / n, total_concept_loss / n, diag_bacc


@torch.no_grad()
def evaluate(
    model: SoftJointCBM,
    loader: DataLoader,
    diag_criterion: nn.Module,
    concept_criterion: Dict[str, nn.Module],
    concept_loss_weight: float,
    device: torch.device,
    threshold: float = 0.5,
    train_observed_states: Dict[str, List[int]] = None,
    concept_outputs: Dict[str, Any] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float, float, np.ndarray, np.ndarray]:
    # Đánh giá đồng thời chẩn đoán (diagnosis) và 7 concept head
    model.eval()
    total_loss = 0.0
    total_diag_loss = 0.0
    total_concept_loss = 0.0
    all_diag_logits: List[torch.Tensor] = []
    all_labels: List[int] = []
    concept_true: Dict[str, List[int]] = {c: [] for c in CONCEPT_NAMES}
    concept_pred: Dict[str, List[int]] = {c: [] for c in CONCEPT_NAMES}
    concept_probabilities = {c: [] for c in CONCEPT_NAMES}

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        concept_indices = batch["concept_indices"].to(device)

        diag_logits, concept_logits, _ = model(images)
        diag_loss = diag_criterion(diag_logits, labels)
        concept_loss = _concept_loss(concept_logits, concept_indices, concept_criterion)
        loss = diag_loss + concept_loss_weight * concept_loss

        bs = len(labels)
        total_loss += loss.item() * bs
        total_diag_loss += diag_loss.item() * bs
        total_concept_loss += concept_loss.item() * bs

        all_diag_logits.append(diag_logits.cpu())
        all_labels.extend(labels.cpu().numpy())

        for i, c_name in enumerate(CONCEPT_NAMES):
            preds_c = torch.argmax(concept_logits[c_name], dim=-1).cpu().numpy()
            concept_pred[c_name].extend(preds_c.tolist())
            concept_true[c_name].extend(concept_indices[:, i].cpu().numpy().tolist())
            if concept_outputs is not None:
                concept_probabilities[c_name].append(torch.softmax(concept_logits[c_name], dim=-1).cpu().numpy())

    all_logits_tensor = torch.cat(all_diag_logits, dim=0)
    probs = torch.softmax(all_logits_tensor, dim=-1)[:, 1].numpy()
    preds = (probs >= threshold).astype(int)
    labels_np = np.array(all_labels, dtype=int)
    n = len(labels_np)

    diag_metrics = compute_metrics(labels_np, preds, probs)
    concept_metrics = compute_concept_metrics(
        {c: np.array(concept_true[c]) for c in CONCEPT_NAMES},
        {c: np.array(concept_pred[c]) for c in CONCEPT_NAMES},
        CONCEPT_NUM_CLASSES,
        train_observed_states=train_observed_states,
    )
    if concept_outputs is not None:
        concept_outputs.update({"true": {c: np.asarray(concept_true[c]) for c in CONCEPT_NAMES},
                                "pred": {c: np.asarray(concept_pred[c]) for c in CONCEPT_NAMES},
                                "probabilities": {c: np.concatenate(concept_probabilities[c]) for c in CONCEPT_NAMES}})
    return (
        diag_metrics,
        concept_metrics,
        total_loss / n,
        total_diag_loss / n,
        total_concept_loss / n,
        preds,
        probs,
    )


def _default_paths(
    seed: int = DEFAULT_SEED,
    augmentation_preset: str = DEFAULT_AUGMENTATION_PRESET,
    concept_loss_weight: float = DEFAULT_CONCEPT_LOSS_WEIGHT,
) -> Tuple[str, str]:
    stem = f"m3_soft_joint_cbm_seed{seed}"
    if augmentation_preset != DEFAULT_AUGMENTATION_PRESET:
        stem = f"m3_soft_joint_cbm_{augmentation_preset}_seed{seed}"
    return (
        os.path.join(PROJECT_ROOT, "checkpoints", f"{stem}_best.pth"),
        os.path.join(PROJECT_ROOT, "results", f"{stem}_results.json"),
    )


def _prepare_output(path: str, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Artifact already exists: {path}. Choose another path or pass --overwrite.")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run_m3_experiment(
    manifest_path: str = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    concept_loss_weight: float = DEFAULT_CONCEPT_LOSS_WEIGHT,
    seed: int = DEFAULT_SEED,
    device_name: str = "auto",
    checkpoint_path: str = None,
    results_path: str = None,
    save_results: bool = True,
    augmentation_preset: str = DEFAULT_AUGMENTATION_PRESET,
    overwrite: bool = False,
    num_workers: int = 2,
) -> Dict[str, Any]:
    """Train and select the checkpoint/threshold using train and validation only."""
    if epochs < 1 or batch_size < 1 or num_workers < 0:
        raise ValueError("epochs/batch_size must be positive and num_workers non-negative")
    if not np.isfinite(concept_loss_weight) or concept_loss_weight < 0:
        raise ValueError("concept_loss_weight must be finite and non-negative")
    if not np.isfinite(lr) or lr <= 0 or not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    set_seed(seed)
    device = get_device(device_name)

    if manifest_path is None:
        manifest_path = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
    manifest_path = os.path.abspath(manifest_path)
    checkpoint_path, results_path = prepare_training_outputs(
        _default_paths(seed, augmentation_preset, concept_loss_weight),
        checkpoint_path, results_path, overwrite, save_results,
    )

    _prepare_output(checkpoint_path, overwrite)
    if save_results:
        _prepare_output(results_path, overwrite)

    print("Khởi động M3 - Soft Joint CBM (EfficientNet-B0 -> 7 concept heads -> g Linear)")
    print(f"Thiết bị sử dụng: {device} | Random seed: {seed}")
    print(
        f"Cấu hình: Epochs={epochs}, Batch Size={batch_size}, LR={lr}, Weight Decay={weight_decay}, "
        f"Augmentation={augmentation_preset}, Concept Loss Weight (lambda)={concept_loss_weight}"
    )

    bundle = get_dataloaders(
        manifest_path=manifest_path,
        project_root=PROJECT_ROOT,
        batch_size=batch_size,
        num_workers=num_workers,
        target_size=224,
        augment_train=True,
        augmentation_preset=augmentation_preset,
        include_test=False,
    )

    train_loader = bundle["train"]
    valid_loader = bundle["valid"]
    class_weights = bundle["class_weights"].to(device)

    assert len(bundle["datasets"]["train"]) == 413, "Train split phải có đúng 413 mẫu"
    assert len(bundle["datasets"]["valid"]) == 203, "Valid split phải có đúng 203 mẫu"

    label_mapping = bundle["datasets"]["train"].label_mapping
    schema = load_concept_schema(manifest_path)
    train_observed_states = compute_train_observed_states(bundle["datasets"]["train"].df, label_mapping)
    statistics = compute_concept_statistics(manifest_path, PROJECT_ROOT)
    state_weights = {c: statistics[c]["weights"] for c in CONCEPT_NAMES}

    model = get_soft_joint_cbm(CONCEPT_NAMES, CONCEPT_NUM_CLASSES, num_classes=2, pretrained=True)
    model.to(device)

    diag_criterion = nn.CrossEntropyLoss(weight=class_weights)
    concept_criterion = build_concept_criteria(state_weights, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_bacc = -1.0
    best_epoch = -1
    history: List[Dict[str, float]] = []
    config = {
        "model": "M3_SoftJointCBM",
        "protocol": M3_PROTOCOL,
        "backbone": "EfficientNet-B0",
        "pretrained_weights": "EfficientNet_B0_Weights.DEFAULT",
        "diagnosis_head": "Linear(28, 2)",
        "concept_heads": "Linear(1280, Ki) per concept",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "concept_loss_weight": concept_loss_weight,
        "seed": seed,
        "device": str(device),
        "diagnosis_loss": "Weighted_CrossEntropy",
        "concept_loss": "Multi-Head_CrossEntropy_balanced_mean",
        "concept_weighting": "balanced",
        "concept_state_weights": state_weights,
        "concept_train_statistics": statistics,
        "dropout": 0.2,
        "class_weights": [float(x) for x in class_weights.cpu().tolist()],
        "augmentation_preset": augmentation_preset,
        "target_size": 224,
        "num_workers": num_workers,
        "scheduler": "CosineAnnealingLR(eta_min=1e-6)",
        "checkpoint_selection": "validation_balanced_accuracy_at_0.5",
        "threshold_selection": "maximize_validation_balanced_accuracy_tie_nearest_0.5",
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "scikit_learn_version": str(sklearn.__version__),
        "platform": platform.platform(),
    }
    manifest_sha256 = _manifest_hash(manifest_path)

    print("\n--- Bắt đầu huấn luyện M3 (Soft Joint CBM) ---")
    for epoch in range(1, epochs + 1):
        train_loss, train_diag_loss, train_concept_loss, train_bacc = train_one_epoch(
            model, train_loader, diag_criterion, concept_criterion, concept_loss_weight, optimizer, device
        )
        val_metrics, val_concept_metrics, val_loss, val_diag_loss, val_concept_loss, _, _ = evaluate(
            model, valid_loader, diag_criterion, concept_criterion, concept_loss_weight, device,
            train_observed_states=train_observed_states,
        )
        scheduler.step()

        val_bacc = val_metrics["balanced_accuracy"]
        val_f1_mel = val_metrics["f1_melanoma"]
        val_auc = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else 0.0
        val_concept_f1 = val_concept_metrics["overall_f1_macro_all_defined"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_diagnosis_loss": train_diag_loss,
            "train_concept_loss": train_concept_loss,
            "train_diagnosis_balanced_accuracy": train_bacc,
            "validation_loss": val_loss,
            "validation_diagnosis_loss": val_diag_loss,
            "validation_concept_loss": val_concept_loss,
            "validation_balanced_accuracy_at_0.5": val_bacc,
            "validation_f1_melanoma_at_0.5": val_f1_mel,
            "validation_roc_auc": val_auc,
            "validation_pr_auc": val_metrics["pr_auc"],
            "validation_concept_f1_macro_all_defined": val_concept_f1,
        })

        is_best = val_bacc > best_val_bacc
        best_marker = " [BEST]" if is_best else ""

        if is_best:
            best_val_bacc = val_bacc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_balanced_acc": val_bacc,
                "val_metrics": val_metrics,
                "val_concept_metrics": val_concept_metrics,
                "config": config,
                "manifest_sha256": manifest_sha256,
                "manifest_fingerprint": manifest_fingerprint(manifest_path),
                "concept_schema": schema,
            }, checkpoint_path)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} (Diag: {train_diag_loss:.4f}, Concept: {train_concept_loss:.4f}, BAcc: {train_bacc * 100:.2f}%) | "
            f"Valid Loss: {val_loss:.4f} | Valid BAcc: {val_bacc * 100:.2f}% | Valid F1-Mel: {val_f1_mel:.4f} | "
            f"Valid AUC: {val_auc:.4f} | Valid Concept F1: {val_concept_f1:.4f}{best_marker}"
        )

    print(f"\nHuấn luyện hoàn tất! Checkpoint tốt nhất tại Epoch {best_epoch:02d} (Valid Balanced Acc: {best_val_bacc * 100:.2f}%)")
    print(f"Checkpoint đã lưu tại: {checkpoint_path}")

    # Chọn threshold trên validation của checkpoint đã chọn, không đọc test.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    load_soft_joint_cbm_state_dict(model, checkpoint["model_state_dict"])
    val_concept_outputs = {}
    (
        val_metrics_at_05, val_concept_metrics_at_05, val_loss_best, _, _, _, val_probs_best,
    ) = evaluate(
        model, valid_loader, diag_criterion, concept_criterion, concept_loss_weight, device,
        train_observed_states=train_observed_states,
        concept_outputs=val_concept_outputs,
    )
    val_true = bundle["datasets"]["valid"].df["diagnosis_binary"].to_numpy(dtype=int)
    threshold, _ = select_validation_threshold(val_true, val_probs_best)
    val_preds_best = (val_probs_best >= threshold).astype(int)
    val_metrics_best = compute_metrics(val_true, val_preds_best, val_probs_best)
    checkpoint["decision_threshold"] = threshold
    checkpoint["validation_metrics_at_threshold"] = val_metrics_best
    torch.save(checkpoint, checkpoint_path)

    valid_table = print_metrics_table(
        val_metrics_best, title=f"Validation Set (Epoch {best_epoch}, threshold={threshold:.6f})"
    )
    concept_table = print_concept_metrics_table(val_concept_metrics_at_05)
    val_sample_preds = _extract_sample_predictions(
        bundle["datasets"]["valid"].df, val_preds_best, val_probs_best, val_concept_outputs
    )

    summary_text = (
        f"M3 Soft Joint CBM (EfficientNet-B0 -> 7 concept heads -> g Linear)\n"
        f"Best Epoch: {best_epoch}/{epochs}\n"
        f"Training samples: 413, Valid: 203\n"
        f"Threshold selected on validation: {threshold:.6f}\n\n"
        f"{valid_table}\n\n{concept_table}"
    )

    results = {
        "model": "M3_SoftJointCBM",
        "mode": "train_validation",
        "best_epoch": int(best_epoch),
        "hyperparameters": config,
        "manifest_sha256": manifest_sha256,
        "manifest_fingerprint": manifest_fingerprint(manifest_path),
        "concept_schema": schema,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "decision_threshold": threshold,
        "summary": summary_text,
        "validation_metrics_at_0.5": val_metrics_at_05,
        "validation_metrics": val_metrics_best,
        "validation_concept_metrics": val_concept_metrics_at_05,
        "validation_loss": val_loss_best,
        "validation_predictions": val_sample_preds,
        "train_observed_states": train_observed_states,
        "history": history,
    }

    if save_results:
        _save_json(results_path, results)
        print(f"Kết quả chi tiết đã lưu tại: {results_path}")

    return results


def evaluate_m3_test(
    checkpoint_path: str,
    manifest_path: str = None,
    device_name: str = "auto",
    results_path: str = None,
    save_results: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """One explicit final test evaluation using a frozen checkpoint and threshold."""
    if manifest_path is None:
        manifest_path = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
    manifest_path = os.path.abspath(manifest_path)
    device = get_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "decision_threshold" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint has no frozen validation threshold/config; use a newly trained checkpoint")
    validate_manifest(checkpoint, manifest_path)
    schema = validate_concept_schema(checkpoint, manifest_path)

    config = checkpoint["config"]
    validate_m3_config(config)
    threshold = float(checkpoint["decision_threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold in checkpoint")
    if results_path is None:
        _, results_path = _default_paths(config["seed"], config["augmentation_preset"], config["concept_loss_weight"])
    if save_results:
        _prepare_output(results_path, overwrite)

    train_df = pd.read_csv(manifest_path, dtype={"is_inconsistent_profile": bool})
    train_df = train_df[train_df["split"] == "train"].reset_index(drop=True)
    label_mapping = schema["label_mapping"]
    train_observed_states = compute_train_observed_states(train_df, label_mapping)

    test_dataset = Derm7ptDataset(
        manifest_path=manifest_path,
        project_root=PROJECT_ROOT,
        split="test",
        label_mapping=label_mapping,
        transform=get_transforms("valid", target_size=config["target_size"], augment=False),
    )
    assert len(test_dataset) == 395, "Test split phải có đúng 395 mẫu"
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    model = get_soft_joint_cbm(schema["concept_names"], schema["concept_num_classes"], num_classes=2,
                              pretrained=False, dropout=config.get("dropout", 0.2)).to(device)
    load_soft_joint_cbm_state_dict(model, checkpoint["model_state_dict"])
    weights = torch.tensor(config["class_weights"], dtype=torch.float32, device=device)
    diag_criterion = nn.CrossEntropyLoss(weight=weights)
    concept_criterion = build_concept_criteria(config.get("concept_state_weights"), device)
    test_concept_outputs = {}

    test_metrics, test_concept_metrics, test_loss, _, _, test_preds, test_probs = evaluate(
        model, test_loader, diag_criterion, concept_criterion, config["concept_loss_weight"], device,
        threshold=threshold, train_observed_states=train_observed_states,
        concept_outputs=test_concept_outputs,
    )
    test_table = print_metrics_table(test_metrics, title=f"Test Set (threshold={threshold:.6f})")
    concept_table = print_concept_metrics_table(test_concept_metrics)

    results = {
        "model": "M3_SoftJointCBM",
        "mode": "final_test",
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": _manifest_hash(checkpoint_path),
        "best_epoch": int(checkpoint["epoch"]),
        "hyperparameters": config,
        "manifest_sha256": checkpoint["manifest_sha256"],
        "manifest_fingerprint": manifest_fingerprint(manifest_path),
        "concept_schema": schema,
        "decision_threshold": threshold,
        "summary": f"{test_table}\n\n{concept_table}",
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "test_concept_metrics": test_concept_metrics,
        "test_predictions": _extract_sample_predictions(test_dataset.df, test_preds, test_probs,
                                                       test_concept_outputs or None),
    }
    if save_results:
        save_experiment_results(results_path, test=results, overwrite=overwrite)
        print(f"Kết quả test đã lưu tại: {results_path}")
    return results


def run_m3_intervention(checkpoint_path, predictions_path, manifest_path=None,
                        results_path=None, overwrite=False):
    """Evaluate all 128 group subsets using the frozen weighted M3 test export."""
    manifest_path = os.path.abspath(manifest_path or str(Path(PROJECT_ROOT) / "data/manifest.csv"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_manifest(checkpoint, manifest_path)
    schema = validate_concept_schema(checkpoint, manifest_path)
    validate_m3_config(checkpoint["config"])
    export = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    validate_manifest(export, manifest_path)
    validate_concept_schema(export, manifest_path)
    if export.get("mode") not in {"final_test", "train_validation_test"} or export.get("model") != "M3_SoftJointCBM":
        raise ValueError("Expected frozen M3 test predictions")
    if export.get("hyperparameters") != checkpoint["config"]:
        raise ValueError("Prediction training config differs from checkpoint")
    if export.get("checkpoint_sha256") != _manifest_hash(checkpoint_path):
        raise ValueError("Prediction export does not belong to this checkpoint; re-export M3 test")
    threshold = float(checkpoint["decision_threshold"])
    if export.get("decision_threshold") != threshold:
        raise ValueError("Prediction threshold differs from frozen checkpoint")
    test = pd.read_csv(manifest_path)
    test = test[test.split == "test"].set_index("case_num")
    rows = export["test_predictions"]
    ids = [int(row["case_num"]) for row in rows]
    if len(ids) != len(test) or set(ids) != set(test.index) or len(set(ids)) != len(ids):
        raise ValueError("Prediction case IDs differ from the official test split")
    probabilities, true_indices, labels = [], [], []
    for row in rows:
        case = test.loc[row["case_num"]]
        if row["y_true"] != int(case.diagnosis_binary):
            raise ValueError("Prediction labels differ from manifest")
        targets = [schema["label_mapping"][c][case[c]] for c in schema["concept_names"]]
        if "concept_probabilities" not in row or "concept_true" not in row:
            raise ValueError("Export has no per-case concepts; re-export test with the updated M3 runner")
        if targets != [row["concept_true"][c] for c in schema["concept_names"]]:
            raise ValueError("Prediction concept targets differ from manifest")
        if any(len(row["concept_probabilities"][c]) != schema["concept_num_classes"][c]
               for c in schema["concept_names"]):
            raise ValueError("Export concept probability block dimensions differ from schema")
        probabilities.append([p for c in schema["concept_names"] for p in row["concept_probabilities"][c]])
        true_indices.append(targets)
        labels.append(row["y_true"])
    head = nn.Linear(schema["total_states"], 2)
    state = checkpoint["model_state_dict"]
    head.load_state_dict({"weight": state["diagnosis_head.weight"], "bias": state["diagnosis_head.bias"]})
    result = evaluate_soft_interventions(head, probabilities, true_indices, labels, schema, threshold, ids)
    if not np.allclose(result["baseline_y_prob"], [row["y_prob"] for row in rows], atol=1e-6, rtol=1e-5):
        raise ValueError("Export concept probabilities do not reproduce baseline diagnosis")
    if result["baseline_y_pred"] != [row["y_pred"] for row in rows]:
        raise ValueError("Export diagnosis predictions do not use the frozen threshold")
    result.update({"model": "M3_SoftJointCBM", "mode": "intervention_analysis",
                   "hyperparameters": checkpoint["config"],
                   "concept_schema": schema, "manifest_fingerprint": manifest_fingerprint(manifest_path),
                   "checkpoint_sha256": _manifest_hash(checkpoint_path),
                   "checkpoint_epoch": int(checkpoint["epoch"]),
                   "evaluation_environment": {"device": "cpu", "platform": platform.platform(),
                                              "torch_version": str(torch.__version__),
                                              "numpy_version": np.__version__,
                                              "scikit_learn_version": sklearn.__version__},
                   "checkpoint_path": os.path.abspath(checkpoint_path),
                   "predictions_path": os.path.abspath(predictions_path)})
    results_path = results_path or str(Path(predictions_path).with_name(Path(predictions_path).stem + "_intervention.json"))
    if Path(results_path).resolve() in {Path(checkpoint_path).resolve(), Path(predictions_path).resolve(), Path(manifest_path).resolve()}:
        raise ValueError("Intervention output must not overwrite its inputs")
    _prepare_output(results_path, overwrite)
    _save_json(results_path, result)
    print(f"Saved {result['num_subsets']} intervention subsets to {results_path}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="M3 weighted Soft Joint CBM: train → validation selection → automatic test; intervention")
    parser.add_argument("--mode", choices=["train", "test", "intervention"], default="train", help="train: train rồi tự động test; test: đánh giá checkpoint; intervention: can thiệp concepts")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--concept_loss_weight", type=float, default=DEFAULT_CONCEPT_LOSS_WEIGHT,
        help="Trọng số lambda cho concept loss trong L = L_diagnosis + lambda * L_concept",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị: auto, mps, cuda, cpu")
    parser.add_argument(
        "--augmentation_preset",
        choices=["legacy_letterbox", "comparison"],
        default=DEFAULT_AUGMENTATION_PRESET,
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None, help="File JSON chứa validation/test; kết quả can thiệp khi --mode intervention")
    parser.add_argument("--predictions_path", type=str, default=None, help="JSON kết quả M3 chứa test_predictions để phân tích intervention")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args(argv)

    if args.mode == "train":
        checkpoint_path, results_path = prepare_training_outputs(
            _default_paths(args.seed, args.augmentation_preset, args.concept_loss_weight),
            args.checkpoint_path, args.results_path,
            args.overwrite, not args.no_save,
        )
        validation = run_m3_experiment(
            manifest_path=args.manifest_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            concept_loss_weight=args.concept_loss_weight,
            seed=args.seed,
            device_name=args.device,
            checkpoint_path=checkpoint_path,
            save_results=False,
            augmentation_preset=args.augmentation_preset,
            overwrite=args.overwrite,
            num_workers=args.num_workers,
        )
        if not args.no_save:
            save_experiment_results(results_path, validation=validation, overwrite=args.overwrite)
        test = evaluate_m3_test(
            checkpoint_path=checkpoint_path, manifest_path=args.manifest_path,
            device_name=args.device,
            save_results=False, overwrite=args.overwrite,
        )
        if not args.no_save:
            save_experiment_results(results_path, validation=validation, test=test, overwrite=True)
    elif args.mode == "test":
        checkpoint_path = args.checkpoint_path or _default_paths(
            args.seed, args.augmentation_preset, args.concept_loss_weight
        )[0]
        results_path = args.results_path or _default_paths(args.seed, args.augmentation_preset, args.concept_loss_weight)[1]
        if not args.no_save:
            _prepare_output(results_path, args.overwrite)
        test = evaluate_m3_test(
            checkpoint_path=checkpoint_path,
            manifest_path=args.manifest_path,
            device_name=args.device,
            save_results=False,
            overwrite=args.overwrite,
        )
        if not args.no_save:
            save_experiment_results(results_path, test=test, overwrite=args.overwrite)
    else:
        if args.no_save:
            parser.error("--no_save is not supported for intervention mode")
        checkpoint_path = args.checkpoint_path or _default_paths(
            args.seed, args.augmentation_preset, args.concept_loss_weight
        )[0]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        validate_m3_config(config)
        predictions_path = args.predictions_path or _default_paths(
            config["seed"], config["augmentation_preset"], config["concept_loss_weight"]
        )[1]
        run_m3_intervention(checkpoint_path, predictions_path, args.manifest_path,
                            args.results_path, args.overwrite)


if __name__ == "__main__":
    main()
