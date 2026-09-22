from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Metrics cấp Diagnosis (Y): phân loại nhị phân Melanoma vs Non-Melanoma
# ---------------------------------------------------------------------------

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

    # Average Precision là cách định nghĩa PR-AUC được dùng trong đề tài.
    auc = None
    pr_auc = None
    if y_prob is not None:
        y_prob = np.asarray(y_prob).astype(float)
        if len(np.unique(y_prob)) > 1 and len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_prob))
        else:
            auc = 0.5
        if len(np.unique(y_true)) > 1:
            pr_auc = float(average_precision_score(y_true, y_prob))

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(prec),
        "f1_melanoma": float(f1_mel),
        "f1_macro": float(f1_macro),
        "roc_auc": auc,
        "pr_auc": pr_auc,
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
    pr_auc_str = f"{metrics['pr_auc']:.4f}" if metrics["pr_auc"] is not None else "N/A"

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
        f"PR-AUC (AP):       {pr_auc_str}",
        f"Confusion Matrix:  TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}",
    ]
    return "\n".join(lines)


def print_metrics_table(metrics: Dict[str, Any], title: str = "Metrics") -> str:
    # In kết quả đánh giá ra console và trả về chuỗi định dạng
    table_str = format_metrics_table(metrics, title=title)
    print(f"\n{table_str}")
    return table_str


# ---------------------------------------------------------------------------
# Metrics cấp Concept (C): Macro-F1 theo từng concept Ci rồi mean qua 7 concept
# Công thức theo EDA: F1_macro(Ci) tính trên state space của Ci,
# rồi Overall = (1/7) * sum(F1_macro(Ci))
# Báo cáo song song: all-defined (28 states) và train-observed (27 states)
# ---------------------------------------------------------------------------

def compute_concept_metrics(
    concept_true: Dict[str, np.ndarray],
    concept_pred: Dict[str, np.ndarray],
    concept_num_classes: Dict[str, int],
    train_observed_states: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, Any]:
    """
    Tính Macro-F1 cho từng concept Ci trên state space tương ứng,
    rồi lấy trung bình cộng qua 7 concepts.

    concept_true: dict tên concept -> mảng nhãn ground-truth (int)
    concept_pred: dict tên concept -> mảng nhãn dự đoán (int)
    concept_num_classes: dict tên concept -> số trạng thái (Ki)
    train_observed_states: dict tên concept -> danh sách index trạng thái có mặt trong train
                           Nếu None, chỉ tính all-defined; nếu có, tính thêm train-observed.
    """
    per_concept = {}

    for c_name in concept_true:
        y_true_c = np.asarray(concept_true[c_name]).astype(int)
        y_pred_c = np.asarray(concept_pred[c_name]).astype(int)
        n_classes = concept_num_classes[c_name]

        # All-defined Macro-F1: tính trên toàn bộ Ki states
        all_labels = list(range(n_classes))
        f1_all = float(f1_score(y_true_c, y_pred_c, labels=all_labels,
                                average="macro", zero_division=0))

        # Accuracy của concept này
        acc_c = float(accuracy_score(y_true_c, y_pred_c))

        entry = {
            "accuracy": acc_c,
            "f1_macro_all_defined": f1_all,
            "num_states_all": n_classes,
        }

        # Train-observed Macro-F1: chỉ tính trên states đã thấy trong train
        if train_observed_states is not None and c_name in train_observed_states:
            obs_labels = sorted(train_observed_states[c_name])
            f1_obs = float(f1_score(y_true_c, y_pred_c, labels=obs_labels,
                                    average="macro", zero_division=0))
            entry["f1_macro_train_observed"] = f1_obs
            entry["num_states_train_observed"] = len(obs_labels)

        per_concept[c_name] = entry

    # Trung bình cộng qua 7 concepts
    n_concepts = len(per_concept)
    overall_f1_all = sum(v["f1_macro_all_defined"] for v in per_concept.values()) / n_concepts
    overall_acc = sum(v["accuracy"] for v in per_concept.values()) / n_concepts

    result = {
        "overall_concept_accuracy": float(overall_acc),
        "overall_f1_macro_all_defined": float(overall_f1_all),
        "per_concept": per_concept,
    }

    if train_observed_states is not None:
        vals = [v["f1_macro_train_observed"] for v in per_concept.values()
                if "f1_macro_train_observed" in v]
        if vals:
            result["overall_f1_macro_train_observed"] = float(sum(vals) / len(vals))

    return result


def format_concept_metrics_table(metrics: Dict[str, Any]) -> str:
    # Định dạng bảng Macro-F1 từng concept
    lines = ["--- Concept Metrics ---"]

    header = f"  {'Concept':<28} {'Acc':>7} {'F1-All':>8}"
    has_obs = "overall_f1_macro_train_observed" in metrics
    if has_obs:
        header += f" {'F1-Obs':>8}"
    lines.append(header)
    lines.append("  " + "-" * len(header.strip()))

    for c_name, vals in metrics["per_concept"].items():
        line = f"  {c_name:<28} {vals['accuracy']*100:>6.2f}% {vals['f1_macro_all_defined']:>8.4f}"
        if has_obs and "f1_macro_train_observed" in vals:
            line += f" {vals['f1_macro_train_observed']:>8.4f}"
        lines.append(line)

    lines.append("  " + "-" * len(header.strip()))
    line = f"  {'MEAN':<28} {metrics['overall_concept_accuracy']*100:>6.2f}% {metrics['overall_f1_macro_all_defined']:>8.4f}"
    if has_obs:
        line += f" {metrics['overall_f1_macro_train_observed']:>8.4f}"
    lines.append(line)

    return "\n".join(lines)


def print_concept_metrics_table(metrics: Dict[str, Any]) -> str:
    # In bảng Macro-F1 concept ra console và trả về chuỗi
    table_str = format_concept_metrics_table(metrics)
    print(f"\n{table_str}")
    return table_str
