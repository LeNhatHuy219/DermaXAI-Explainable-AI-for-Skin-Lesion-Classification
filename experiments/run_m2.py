import argparse
import hashlib
import json
import os
import platform
import sys
from typing import Any, Dict, List, Tuple

# Thêm thư mục gốc vào đường dẫn hệ thống
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# QUAN TRỌNG: import src.dataset (kéo theo torch qua src/__init__.py) TRƯỚC sklearn.
# Trên một số máy Windows CPU-only, nạp scikit-learn trước rồi mới nạp torch/torchvision
# trong cùng một process gây xung đột thứ tự nạp DLL OpenMP/MKL dẫn đến segfault ngay
# khi torch.quantization được import. Nạp theo thứ tự torch-trước-sklearn-sau tránh
# được lỗi này hoàn toàn. M2 tự nó không cần CNN, nhưng vẫn giữ nguyên thứ tự này để
# an toàn trên mọi máy.
from src.dataset import CONCEPT_NAMES, CONCEPT_OFFSETS, TOTAL_CONCEPT_STATES

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression

from src.metrics import compute_metrics, print_metrics_table


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

# M2 - Oracle Concept Model: 7 concept Ground Truth (one-hot 28 chiều) -> Diagnosis.
# Mục đích: đo trần thông tin (information ceiling) mà 7 concept Derm7pt thực sự chứa
# được cho bài toán chẩn đoán, dùng Logistic Regression - một bộ phân loại g tuyến tính
# đơn giản giống hệt kiến trúc của đầu chẩn đoán trong M3 (Soft Joint CBM) để hai mô hình
# có thể so sánh công bằng (M2 dùng concept Ground Truth, M3 dùng concept dự đoán từ ảnh).

DEFAULT_SEED = 42
DEFAULT_C_GRID: List[float] = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def _default_paths(seed: int = DEFAULT_SEED) -> Tuple[str, str, str]:
    stem = "m2_oracle_lr" if seed == DEFAULT_SEED else f"m2_oracle_lr_seed{seed}"
    return (
        os.path.join(PROJECT_ROOT, "checkpoints", f"{stem}_best.joblib"),
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


def _load_label_mapping(project_root: str) -> Dict[str, Dict[str, int]]:
    mapping_path = os.path.join(project_root, "data", "label_mapping.json")
    assert os.path.exists(mapping_path), f"Thiếu file label_mapping.json tại {mapping_path}"
    with open(mapping_path, "r") as handle:
        return json.load(handle)


def build_oracle_features(df: pd.DataFrame, label_mapping: Dict[str, Dict[str, int]]) -> np.ndarray:
    """Ghép 7 concept Ground Truth thành vector one-hot 28 chiều (giống concept_onehot)."""
    n = len(df)
    features = np.zeros((n, TOTAL_CONCEPT_STATES), dtype=np.float32)
    for row_pos, (_, row) in enumerate(df.iterrows()):
        for c_name in CONCEPT_NAMES:
            c_idx = label_mapping[c_name][row[c_name]]
            features[row_pos, CONCEPT_OFFSETS[c_name] + c_idx] = 1.0
    return features


def concept_state_names(label_mapping: Dict[str, Dict[str, int]]) -> List[str]:
    # Tên đầy đủ "concept=state" cho từng cột của vector 28 chiều, phục vụ diễn giải hệ số hồi quy
    names = [""] * TOTAL_CONCEPT_STATES
    for c_name in CONCEPT_NAMES:
        for state_name, state_idx in label_mapping[c_name].items():
            names[CONCEPT_OFFSETS[c_name] + state_idx] = f"{c_name}={state_name}"
    return names


def _extract_sample_predictions(df: pd.DataFrame, y_pred: np.ndarray, y_prob: np.ndarray) -> List[Dict[str, Any]]:
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


def _select_best_C(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    c_grid: List[float],
    seed: int,
) -> Tuple[float, Dict[float, float], LogisticRegression]:
    """Chọn C tối đa Balanced Accuracy trên validation tại ngưỡng 0.5 (giống lựa chọn checkpoint của M1)."""
    scores: Dict[float, float] = {}
    best_c = c_grid[0]
    best_score = -1.0
    best_model = None
    for c in c_grid:
        model = LogisticRegression(
            C=c, class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=seed,
        )
        model.fit(X_train, y_train)
        valid_probs = model.predict_proba(X_valid)[:, 1]
        valid_preds = (valid_probs >= 0.5).astype(int)
        bacc = compute_metrics(y_valid, valid_preds)["balanced_accuracy"]
        scores[c] = bacc
        if bacc > best_score + 1e-12:
            best_score = bacc
            best_c = c
            best_model = model
    return best_c, scores, best_model


def run_m2_experiment(
    manifest_path: str = None,
    seed: int = DEFAULT_SEED,
    c_grid: List[float] = None,
    checkpoint_path: str = None,
    results_path: str = None,
    save_results: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Chọn hyperparameter C và ngưỡng quyết định trên train/validation. Test là bước riêng."""
    if manifest_path is None:
        manifest_path = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
    manifest_path = os.path.abspath(manifest_path)
    if c_grid is None:
        c_grid = DEFAULT_C_GRID

    default_checkpoint, default_validation, _ = _default_paths(seed)
    if checkpoint_path is None:
        checkpoint_path = default_checkpoint
    if results_path is None:
        results_path = default_validation

    _prepare_output(checkpoint_path, overwrite)
    if save_results:
        _prepare_output(results_path, overwrite)

    print("Khởi động M2 - Oracle Concept Model (Logistic Regression trên concept Ground Truth)")

    df = pd.read_csv(manifest_path, dtype={"is_inconsistent_profile": bool})
    label_mapping = _load_label_mapping(PROJECT_ROOT)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    valid_df = df[df["split"] == "valid"].reset_index(drop=True)

    assert len(train_df) == 413, "Train split phải có đúng 413 mẫu"
    assert len(valid_df) == 203, "Valid split phải có đúng 203 mẫu"

    X_train = build_oracle_features(train_df, label_mapping)
    y_train = train_df["diagnosis_binary"].to_numpy(dtype=int)
    X_valid = build_oracle_features(valid_df, label_mapping)
    y_valid = valid_df["diagnosis_binary"].to_numpy(dtype=int)

    print(f"Grid search C trên validation: {c_grid}")
    best_c, c_scores, best_model = _select_best_C(X_train, y_train, X_valid, y_valid, c_grid, seed)
    print(f"C tốt nhất theo Balanced Accuracy (threshold=0.5) trên validation: {best_c} (BAcc={c_scores[best_c] * 100:.2f}%)")

    valid_probs = best_model.predict_proba(X_valid)[:, 1]
    valid_metrics_at_05 = compute_metrics(y_valid, (valid_probs >= 0.5).astype(int), valid_probs)

    threshold, _ = select_validation_threshold(y_valid, valid_probs)
    valid_preds = (valid_probs >= threshold).astype(int)
    valid_metrics = compute_metrics(y_valid, valid_preds, valid_probs)
    valid_table = print_metrics_table(valid_metrics, title=f"Validation Set (C={best_c}, threshold={threshold:.6f})")

    manifest_sha256 = _manifest_hash(manifest_path)
    config = {
        "model_type": "LogisticRegression",
        "feature_space": "concept_onehot_28d_ground_truth",
        "class_weight": "balanced",
        "solver": "lbfgs",
        "max_iter": 2000,
        "seed": seed,
        "c_grid": c_grid,
        "best_C": best_c,
        "c_grid_validation_balanced_accuracy": c_scores,
        "checkpoint_selection": "validation_balanced_accuracy_at_0.5",
        "threshold_selection": "maximize_validation_balanced_accuracy_tie_nearest_0.5",
        "scikit_learn_version": str(sklearn.__version__),
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "platform": platform.platform(),
    }

    concept_names_28 = concept_state_names(label_mapping)
    coefficients = {
        name: float(coef) for name, coef in zip(concept_names_28, best_model.coef_.ravel())
    }

    joblib.dump({
        "model": best_model,
        "config": config,
        "manifest_sha256": manifest_sha256,
        "decision_threshold": threshold,
        "validation_metrics_at_threshold": valid_metrics,
        "concept_state_coefficients": coefficients,
        "intercept": float(best_model.intercept_[0]),
    }, checkpoint_path)
    print(f"Checkpoint đã lưu tại: {checkpoint_path}")

    val_sample_preds = _extract_sample_predictions(valid_df, valid_preds, valid_probs)
    summary_text = (
        f"M2 Oracle Concept Model (Logistic Regression, concept Ground Truth)\n"
        f"Training samples: 413, Valid: 203\n"
        f"Best C (validation BAcc@0.5): {best_c}\n"
        f"Threshold selected on validation: {threshold:.6f}\n\n"
        f"{valid_table}"
    )

    results = {
        "model": "M2_Oracle_LogisticRegression",
        "mode": "train_validation",
        "hyperparameters": config,
        "manifest_sha256": manifest_sha256,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "decision_threshold": threshold,
        "summary": summary_text,
        "validation_metrics_at_0.5": valid_metrics_at_05,
        "validation_metrics": valid_metrics,
        "validation_predictions": val_sample_preds,
        "concept_state_coefficients": coefficients,
        "intercept": float(best_model.intercept_[0]),
    }

    if save_results:
        _save_json(results_path, results)
        print(f"Kết quả chi tiết đã lưu tại: {results_path}")

    return results


def evaluate_m2_test(
    checkpoint_path: str,
    manifest_path: str = None,
    results_path: str = None,
    save_results: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Một lần đánh giá test duy nhất bằng model + ngưỡng đã đóng băng."""
    if manifest_path is None:
        manifest_path = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
    manifest_path = os.path.abspath(manifest_path)

    checkpoint = joblib.load(checkpoint_path)
    if "decision_threshold" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint has no frozen validation threshold/config; use a newly trained checkpoint")
    if checkpoint.get("manifest_sha256") != _manifest_hash(manifest_path):
        raise ValueError("The manifest differs from the one used for training")

    config = checkpoint["config"]
    threshold = float(checkpoint["decision_threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold in checkpoint")

    if results_path is None:
        _, _, results_path = _default_paths(config["seed"])
    if save_results:
        _prepare_output(results_path, overwrite)

    df = pd.read_csv(manifest_path, dtype={"is_inconsistent_profile": bool})
    label_mapping = _load_label_mapping(PROJECT_ROOT)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    assert len(test_df) == 395, "Test split phải có đúng 395 mẫu"

    X_test = build_oracle_features(test_df, label_mapping)
    y_test = test_df["diagnosis_binary"].to_numpy(dtype=int)

    model: LogisticRegression = checkpoint["model"]
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)
    test_metrics = compute_metrics(y_test, test_preds, test_probs)
    test_table = print_metrics_table(test_metrics, title=f"Test Set (threshold={threshold:.6f})")

    results = {
        "model": "M2_Oracle_LogisticRegression",
        "mode": "final_test",
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "hyperparameters": config,
        "manifest_sha256": checkpoint["manifest_sha256"],
        "decision_threshold": threshold,
        "summary": test_table,
        "test_metrics": test_metrics,
        "test_predictions": _extract_sample_predictions(test_df, test_preds, test_probs),
    }

    if save_results:
        _save_json(results_path, results)
        print(f"Kết quả test đã lưu tại: {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M2 Oracle Concept Model: train/validation, then explicit final test")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Cho phép ghi đè artifact của cùng run")
    parser.add_argument("--no_save", action="store_true", help="Không lưu kết quả vào file")
    args = parser.parse_args()

    if args.mode == "train":
        run_m2_experiment(
            manifest_path=args.manifest_path,
            seed=args.seed,
            checkpoint_path=args.checkpoint_path,
            results_path=args.results_path,
            save_results=not args.no_save,
            overwrite=args.overwrite,
        )
    else:
        checkpoint_path = args.checkpoint_path or _default_paths(args.seed)[0]
        evaluate_m2_test(
            checkpoint_path=checkpoint_path,
            manifest_path=args.manifest_path,
            results_path=args.results_path,
            save_results=not args.no_save,
            overwrite=args.overwrite,
        )
