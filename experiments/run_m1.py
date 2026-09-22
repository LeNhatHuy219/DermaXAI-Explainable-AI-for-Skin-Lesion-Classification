import argparse
import hashlib
import json
import os
import platform
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

# Thêm thư mục gốc vào đường dẫn hệ thống
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import Derm7ptDataset, get_dataloaders
from src.metrics import compute_metrics, print_metrics_table
from src.models import BlackBoxClassifier, load_blackbox_state_dict
from src.transforms import get_transforms

# Các tham số huấn luyện mặc định
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_SEED = 42
DEFAULT_AUGMENTATION_PRESET = "legacy_letterbox"


def set_seed(seed: int = DEFAULT_SEED) -> None:
    # Cố định seed cho các thư viện để kết quả có thể tái lập
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(device_arg: str = "auto") -> torch.device:
    # Tự động ưu tiên CUDA (NVIDIA GPU / Colab / Server), sau đó Apple Silicon (MPS) và CPU
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _extract_sample_predictions(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> List[Dict[str, Any]]:
    # Trích xuất chi tiết dự đoán từng ca bệnh phục vụ phân tích lỗi
    records = []
    for (_, row), pred, prob in zip(df.iterrows(), y_pred, y_prob):
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
    return records


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    # Huấn luyện mô hình trong 1 epoch và trả về (train_loss, train_balanced_acc)
    model.train()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics["balanced_accuracy"]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[Dict[str, Any], float, np.ndarray, np.ndarray]:
    # Đánh giá mô hình trên tập validation hoặc test
    model.eval()
    total_loss = 0.0
    all_logits: List[torch.Tensor] = []
    all_labels: List[int] = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        all_logits.append(logits.cpu())
        all_labels.extend(labels.cpu().numpy())

    all_logits_tensor = torch.cat(all_logits, dim=0)
    probs = torch.softmax(all_logits_tensor, dim=-1)[:, 1].numpy()
    preds = (probs >= threshold).astype(int)
    labels_np = np.array(all_labels, dtype=int)

    avg_loss = total_loss / len(labels_np)
    metrics = compute_metrics(labels_np, preds, probs)
    return metrics, avg_loss, preds, probs


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


def _manifest_hash(manifest_path: str) -> str:
    digest = hashlib.sha256()
    with open(manifest_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_paths(seed: int = DEFAULT_SEED, augmentation_preset: str = DEFAULT_AUGMENTATION_PRESET) -> Tuple[str, str, str]:
    if seed == DEFAULT_SEED and augmentation_preset == DEFAULT_AUGMENTATION_PRESET:
        stem = "m1_efficientnet_b0"
    else:
        stem = f"m1_efficientnet_b0_{augmentation_preset}_seed{seed}"
    return (
        os.path.join(PROJECT_ROOT, "checkpoints", f"{stem}_best.pth"),
        os.path.join(PROJECT_ROOT, "results", f"{stem}_validation.json"),
        os.path.join(PROJECT_ROOT, "results", f"{stem}_test.json"),
    )


def _prepare_output(path: str, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Artifact already exists: {path}. Choose another path or pass --overwrite.")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run_m1_experiment(
    manifest_path: str = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_SEED,
    device_name: str = "auto",
    checkpoint_path: str = None,
    results_path: str = None,
    save_results: bool = True,
    augmentation_preset: str = DEFAULT_AUGMENTATION_PRESET,
    overwrite: bool = False,
    num_workers: int = 2,
) -> Dict[str, Any]:
    """Train/select on train+validation only. Test evaluation is a separate command."""
    if epochs < 1 or batch_size < 1 or num_workers < 0:
        raise ValueError("epochs/batch_size must be positive and num_workers non-negative")
    set_seed(seed)
    device = get_device(device_name)

    if manifest_path is None:
        manifest_path = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
    manifest_path = os.path.abspath(manifest_path)
    default_checkpoint, default_validation, _ = _default_paths(seed, augmentation_preset)
    if checkpoint_path is None:
        checkpoint_path = default_checkpoint
    if results_path is None:
        results_path = default_validation

    _prepare_output(checkpoint_path, overwrite)
    if save_results:
        _prepare_output(results_path, overwrite)

    print("Khởi động M1 - Black-box EfficientNet-B0 (train/validation only)")
    print(f"Thiết bị sử dụng: {device} | Random seed: {seed}")
    print(f"Cấu hình: Epochs={epochs}, Batch Size={batch_size}, LR={lr}, Weight Decay={weight_decay}, Augmentation={augmentation_preset}")

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

    model = BlackBoxClassifier(num_classes=2, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_bacc = -1.0
    best_epoch = -1
    history: List[Dict[str, float]] = []
    config = {
        "backbone": "EfficientNet-B0",
        "pretrained_weights": "EfficientNet_B0_Weights.DEFAULT",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "device": str(device),
        "loss": "Weighted_CrossEntropy",
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

    print("\n--- Bắt đầu huấn luyện M1 (EfficientNet-B0) ---")
    for epoch in range(1, epochs + 1):
        train_loss, train_bacc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_loss, _, _ = evaluate(model, valid_loader, criterion, device)
        scheduler.step()

        val_bacc = val_metrics["balanced_accuracy"]
        val_f1_mel = val_metrics["f1_melanoma"]
        val_auc = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else 0.0
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_balanced_accuracy": train_bacc,
            "validation_loss": val_loss,
            "validation_balanced_accuracy_at_0.5": val_bacc,
            "validation_f1_melanoma_at_0.5": val_f1_mel,
            "validation_roc_auc": val_auc,
            "validation_pr_auc": val_metrics["pr_auc"],
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
                "config": config,
                "manifest_sha256": manifest_sha256,
            }, checkpoint_path)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} (BAcc: {train_bacc * 100:.2f}%) | "
            f"Valid Loss: {val_loss:.4f} | "
            f"Valid BAcc: {val_bacc * 100:.2f}% | "
            f"Valid F1-Mel: {val_f1_mel:.4f} | "
            f"Valid AUC: {val_auc:.4f}{best_marker}"
        )

    print(f"\nHuấn luyện hoàn tất! Checkpoint tốt nhất tại Epoch {best_epoch:02d} (Valid Balanced Acc: {best_val_bacc * 100:.2f}%)")
    print(f"Checkpoint đã lưu tại: {checkpoint_path}")

    # Chọn threshold trên validation của checkpoint đã chọn, không đọc test.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    load_blackbox_state_dict(model, checkpoint["model_state_dict"])
    val_metrics_at_05, val_loss_best, _, val_probs_best = evaluate(model, valid_loader, criterion, device)
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
    val_sample_preds = _extract_sample_predictions(
        bundle["datasets"]["valid"].df, val_preds_best, val_probs_best
    )

    summary_text = (
        f"M1 Black-box Baseline (EfficientNet-B0)\n"
        f"Best Epoch: {best_epoch}/{epochs}\n"
        f"Training samples: 413, Valid: 203\n"
        f"Threshold selected on validation: {threshold:.6f}\n\n"
        f"{valid_table}"
    )

    results = {
        "model": "M1_BlackBox_EfficientNetB0",
        "mode": "train_validation",
        "best_epoch": int(best_epoch),
        "hyperparameters": config,
        "manifest_sha256": manifest_sha256,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "decision_threshold": threshold,
        "summary": summary_text,
        "validation_metrics_at_0.5": val_metrics_at_05,
        "validation_metrics": val_metrics_best,
        "validation_loss": val_loss_best,
        "validation_predictions": val_sample_preds,
        "history": history,
    }

    if save_results:
        _save_json(results_path, results)
        print(f"Kết quả chi tiết đã lưu tại: {results_path}")

    return results


def evaluate_m1_test(
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
    if checkpoint.get("manifest_sha256") != _manifest_hash(manifest_path):
        raise ValueError("The manifest differs from the one used for training")

    config = checkpoint["config"]
    threshold = float(checkpoint["decision_threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold in checkpoint")
    if results_path is None:
        _, _, results_path = _default_paths(config["seed"], config["augmentation_preset"])
    if save_results:
        _prepare_output(results_path, overwrite)

    test_dataset = Derm7ptDataset(
        manifest_path=manifest_path,
        project_root=PROJECT_ROOT,
        split="test",
        transform=get_transforms("valid", target_size=config["target_size"], augment=False),
    )
    assert len(test_dataset) == 395, "Test split phải có đúng 395 mẫu"
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    model = BlackBoxClassifier(num_classes=2, pretrained=False).to(device)
    load_blackbox_state_dict(model, checkpoint["model_state_dict"])
    weights = torch.tensor(config["class_weights"], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    test_metrics, test_loss, test_preds, test_probs = evaluate(
        model, test_loader, criterion, device, threshold=threshold
    )
    test_table = print_metrics_table(test_metrics, title=f"Test Set (threshold={threshold:.6f})")
    results = {
        "model": "M1_BlackBox_EfficientNetB0",
        "mode": "final_test",
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "best_epoch": int(checkpoint["epoch"]),
        "hyperparameters": config,
        "manifest_sha256": checkpoint["manifest_sha256"],
        "decision_threshold": threshold,
        "summary": test_table,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "test_predictions": _extract_sample_predictions(test_dataset.df, test_preds, test_probs),
    }
    if save_results:
        _save_json(results_path, results)
        print(f"Kết quả test đã lưu tại: {results_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M1 EfficientNet-B0: train/validation, then explicit final test")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Số epoch huấn luyện (mặc định: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Kích thước batch (mặc định: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help=f"Tốc độ học (mặc định: {DEFAULT_LR})")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (mặc định: {DEFAULT_SEED})")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị: auto, mps, cuda, cpu")
    parser.add_argument(
        "--augmentation_preset",
        choices=["legacy_letterbox", "comparison"],
        default=DEFAULT_AUGMENTATION_PRESET,
        help=f"Preset tăng cường dữ liệu: legacy_letterbox (giữ tỷ lệ ảnh) hoặc comparison (mặc định: {DEFAULT_AUGMENTATION_PRESET})",
    )
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers khi train; dùng 0 trong notebook nếu macOS spawn gặp lỗi")
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Cho phép ghi đè artifact của cùng run")
    parser.add_argument("--no_save", action="store_true", help="Không lưu kết quả vào file")
    args = parser.parse_args()

    if args.mode == "train":
        run_m1_experiment(
            manifest_path=args.manifest_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device_name=args.device,
            checkpoint_path=args.checkpoint_path,
            results_path=args.results_path,
            save_results=not args.no_save,
            augmentation_preset=args.augmentation_preset,
            overwrite=args.overwrite,
            num_workers=args.num_workers,
        )
    else:
        checkpoint_path = args.checkpoint_path or _default_paths(args.seed, args.augmentation_preset)[0]
        evaluate_m1_test(
            checkpoint_path=checkpoint_path,
            manifest_path=args.manifest_path,
            device_name=args.device,
            results_path=args.results_path,
            save_results=not args.no_save,
            overwrite=args.overwrite,
        )
