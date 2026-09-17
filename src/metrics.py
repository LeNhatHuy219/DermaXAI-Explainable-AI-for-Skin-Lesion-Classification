from typing import Any, Dict, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob: Optional[Union[np.ndarray, list]] = None,
) -> Dict[str, Any]:
    # Tính các chỉ số đánh giá lâm sàng: Balanced Accuracy, Sensitivity, Specificity, F1, ROC-AUC
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    # Ma trận nhầm lẫn
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Độ nhạy (tỷ lệ phát hiện Melanoma) và độ đặc hiệu (tỷ lệ đúng Non-Melanoma)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_mel = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Tính ROC-AUC nếu có xác suất dự đoán
    auc = None
    if y_prob is not None:
        y_prob = np.asarray(y_prob).astype(float)
        if len(np.unique(y_prob)) > 1 and len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_prob))
        else:
            auc = 0.5

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(prec),
        "f1_melanoma": float(f1_mel),
        "f1_macro": float(f1_macro),
        "roc_auc": auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "total_samples": int(len(y_true)),
        "pos_samples": int(tp + fn),
        "neg_samples": int(tn + fp),
    }


def format_metrics_table(metrics: Dict[str, Any], title: str = "Metrics") -> str:
    # Định dạng chuỗi hiển thị bảng chỉ số đánh giá
    cm = metrics["confusion_matrix"]
    auc_str = f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "N/A"

    lines = [
        f"--- {title} ---",
        f"Total:             {metrics['total_samples']} (Class 0: {metrics['neg_samples']}, Class 1: {metrics['pos_samples']})",
        f"Accuracy:          {metrics['accuracy'] * 100:.2f}%",
        f"Balanced Accuracy: {metrics['balanced_accuracy'] * 100:.2f}%",
        f"Sensitivity:       {metrics['sensitivity'] * 100:.2f}%",
        f"Specificity:       {metrics['specificity'] * 100:.2f}%",
        f"Precision:         {metrics['precision'] * 100:.2f}%",
        f"F1 (Melanoma):     {metrics['f1_melanoma']:.4f}",
        f"F1 (Macro):        {metrics['f1_macro']:.4f}",
        f"ROC-AUC:           {auc_str}",
        f"Confusion Matrix:  TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}",
    ]
    return "\n".join(lines)


def print_metrics_table(metrics: Dict[str, Any], title: str = "Metrics") -> str:
    # In kết quả đánh giá ra console và trả về chuỗi định dạng
    table_str = format_metrics_table(metrics, title=title)
    print(f"\n{table_str}")
    return table_str
