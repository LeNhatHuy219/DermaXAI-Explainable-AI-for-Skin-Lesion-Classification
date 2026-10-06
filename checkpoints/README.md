# Checkpoints theo tiêu chí chọn epoch

| Thư mục | Tiêu chí chọn checkpoint trên validation @0.5 | Weights |
|---|---|---:|
| [bacc/](bacc/manifest.json) | Balanced Accuracy, bộ lịch sử và các studies BAcc | 34 |
| [f1_macro/](f1_macro/README.md) | Diagnosis Macro F1, bộ kết quả chính hiện tại | 25 |

Các runners M1–M4 mặc định dùng `--checkpoint_metric f1_macro` và lưu trong `f1_macro/`. Truyền `--checkpoint_metric balanced_accuracy` để tìm/lưu bộ chính trong `bacc/baseline/`; các studies BAcc có thư mục riêng bên trong `bacc/`. `--checkpoint_path` tùy chỉnh vẫn được ưu tiên.

Tên file F1 giữ hậu tố `_ckptf1macro_best.pth` hoặc `_ckptf1macro_best.joblib`. Ngưỡng dự đoán được chọn riêng trên validation bằng BAcc, rồi đóng băng khi test/intervention; ngưỡng này không quyết định thư mục checkpoint.

Đợt sắp xếp ngày 05/10/2026 chuyển nguyên 25 weights F1 từ tầng gốc vào `f1_macro/` và cập nhật đường dẫn trong JSON kết quả. SHA-256 của weights, metrics, predictions, epochs và thresholds giữ nguyên. Git chỉ lưu README và manifest trong `checkpoints/`; các file weights tiếp tục được bỏ qua.
