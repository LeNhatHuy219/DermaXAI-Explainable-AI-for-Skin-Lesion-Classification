import json
import os
import sys
import numpy as np
import pandas as pd

# Thêm thư mục gốc vào đường dẫn hệ thống
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.metrics import compute_metrics, print_metrics_table


def _extract_sample_predictions(subset_df: pd.DataFrame, y_pred: np.ndarray, y_prob: np.ndarray) -> list:
    # Trích xuất kết quả dự đoán chi tiết từng mẫu phục vụ phân tích lỗi
    records = []
    for (_, row), pred, prob in zip(subset_df.iterrows(), y_pred, y_prob):
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


def run_majority_baseline(manifest_path: str = None, save_results: bool = True):
    # Mô hình cơ sở M0: luôn dự đoán lớp đa số (Non-Melanoma) làm mốc sàn đánh giá
    if manifest_path is None:
        manifest_path = os.path.join(PROJECT_ROOT, "data", "manifest.csv")

    assert os.path.exists(manifest_path), f"Không tìm thấy file manifest: {manifest_path}"

    df = pd.read_csv(manifest_path)

    # Tách dữ liệu theo 3 tập chính thức
    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    test_df = df[df["split"] == "test"].copy()

    # Kiểm tra tính toàn vẹn của dữ liệu và phân chia (sanity check)
    assert len(train_df) == 413, f"Số mẫu train không khớp chuẩn Derm7pt: {len(train_df)} != 413"
    assert len(valid_df) == 203, f"Số mẫu valid không khớp chuẩn Derm7pt: {len(valid_df)} != 203"
    assert len(test_df) == 395, f"Số mẫu test không khớp chuẩn Derm7pt: {len(test_df)} != 395"

    train_labels = train_df["diagnosis_binary"].values
    valid_labels = valid_df["diagnosis_binary"].values
    test_labels = test_df["diagnosis_binary"].values

    assert set(np.unique(train_labels)).issubset({0, 1}), "Nhãn chẩn đoán phải thuộc tập nhị phân {0, 1}"

    # Xác định lớp chiếm đa số và xác suất tiên nghiệm của Melanoma trên tập train
    n_train_neg = int((train_labels == 0).sum())
    n_train_pos = int((train_labels == 1).sum())
    majority_class = 0 if n_train_neg >= n_train_pos else 1
    assert majority_class == 0, f"Lớp đa số tập train phải là 0 (Non-Melanoma), nhận được: {majority_class}"

    melanoma_prior = float(n_train_pos / len(train_labels))

    # Tạo bản tóm tắt dạng text (hiển thị terminal + lưu vào json)
    header = (
        f"Training samples: {len(train_labels)} (Class 0: {n_train_neg}, Class 1: {n_train_pos})\n"
        f"Majority class: {majority_class} (Non-Melanoma), Melanoma Prior: {melanoma_prior:.4f}"
    )
    print(header)

    # Đánh giá trên tập validation
    valid_pred = np.full(len(valid_labels), majority_class)
    valid_prob = np.full(len(valid_labels), melanoma_prior)
    valid_metrics = compute_metrics(valid_labels, valid_pred, valid_prob)
    valid_sample_preds = _extract_sample_predictions(valid_df, valid_pred, valid_prob)
    valid_table = print_metrics_table(valid_metrics, title="Validation Set")

    # Đánh giá trên tập test
    test_pred = np.full(len(test_labels), majority_class)
    test_prob = np.full(len(test_labels), melanoma_prior)
    test_metrics = compute_metrics(test_labels, test_pred, test_prob)
    test_sample_preds = _extract_sample_predictions(test_df, test_pred, test_prob)
    test_table = print_metrics_table(test_metrics, title="Test Set")

    # Ghép toàn bộ nội dung terminal thành chuỗi tóm tắt
    summary = f"{header}\n\n{valid_table}\n\n{test_table}"

    results = {
        "model": "M0_Majority_Baseline",
        "majority_class": majority_class,
        "melanoma_prior": melanoma_prior,
        "summary": summary,
        "validation_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "test_predictions": test_sample_preds,
    }

    # Lưu kết quả vào file json
    if save_results:
        results_dir = os.path.join(PROJECT_ROOT, "results")
        os.makedirs(results_dir, exist_ok=True)
        json_out_path = os.path.join(results_dir, "m0_majority_results.json")
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nKết quả đã lưu tại: {json_out_path}")

    return results


if __name__ == "__main__":
    run_majority_baseline()
