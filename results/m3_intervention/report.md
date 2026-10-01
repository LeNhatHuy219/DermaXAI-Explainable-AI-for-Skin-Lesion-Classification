# Kết quả concept intervention của M3

Seeds: [42, 123, 2026]; 395 ca test; 128 subsets cho mỗi seed.

Thay toàn bộ group được sửa bằng ground-truth one-hot. Giữ nguyên các groups chưa sửa, diagnosis head và threshold validation của từng seed; không huấn luyện lại.

Metrics được tính riêng cho mỗi subset trên toàn bộ test, rồi lấy trung bình theo m, sau đó tổng hợp mean ± sample SD giữa seeds. SD này không phải khoảng tin cậy theo mẫu.

| Số groups sửa | Balanced Accuracy (%) | Macro-F1 | Ca tốt lên (TB) | Ca xấu đi (TB) |
|---|---:|---:|---:|---:|
| 0 | 70.30 ± 4.58 | 0.6844 ± 0.0326 | 0.00 | 0.00 |
| 1 | 61.89 ± 3.85 | 0.6057 ± 0.0333 | 34.52 | 58.24 |
| 2 | 57.74 ± 3.13 | 0.5680 ± 0.0246 | 44.62 | 78.75 |
| 3 | 55.03 ± 2.85 | 0.5426 ± 0.0195 | 48.79 | 90.16 |
| 4 | 52.90 ± 2.82 | 0.5220 ± 0.0187 | 51.15 | 97.74 |
| 5 | 51.09 ± 2.09 | 0.5037 ± 0.0127 | 52.08 | 102.30 |
| 6 | 50.66 ± 0.95 | 0.4995 ± 0.0036 | 53.95 | 103.48 |
| 7 | 53.19 ± 2.83 | 0.5262 ± 0.0243 | 59.67 | 94.67 |

![Intervention curve](intervention_curve.png)

Ca tốt lên/xấu đi được so với m=0 của cùng seed; ở m=1…6 là trung bình qua subsets và seeds, không phải số ca duy nhất. Ở m=7, mỗi seed có một subset sửa đủ bảy groups.

| Sửa toàn bộ 7 groups | Balanced Accuracy (%) | Macro-F1 | Tốt lên | Xấu đi |
|---|---:|---:|---:|---:|
| Seed 42 | 50.75 | 0.5026 | 58 | 82 |
| Seed 123 | 56.29 | 0.5511 | 56 | 101 |
| Seed 2026 | 52.54 | 0.5248 | 65 | 101 |

Sửa đủ bảy groups làm Balanced Accuracy trung bình thay đổi -17.11 điểm phần trăm (70.30% → 53.19%). Đây là kết quả mô tả trên các checkpoints hiện có.

Diagnosis head M3 được học từ soft probabilities; khi intervention, input của head chuyển sang ground-truth one-hot ở các groups được sửa. Sự khác biệt representation là một giả thuyết cần khảo sát để giải thích kết quả, chưa được xác lập là nguyên nhân. Các số liệu này chưa tự chứng minh concept leakage, ảnh hưởng của inconsistency hoặc ưu thế của ECBM.

Các file đi kèm: `summary.json`, `curve.csv`, `per_concept.csv`, `full_intervention_cases.csv` và JSON chi tiết cho từng seed.

`per_concept.csv` đo tác động sửa riêng từng group, không đo tầm quan trọng tổng quát hoặc quan hệ nhân quả. Với M3, groups chưa sửa giữ nguyên dự đoán nên accuracy của chúng không thay đổi khi giữ cùng group. Trung bình qua các group còn lại có thể thay đổi theo subset; tại m=7 chỉ số này không áp dụng.

Đây là mô phỏng hiệu chỉnh annotation. Kết quả sửa toàn bộ concepts dùng diagnosis head M3 đã học và không đồng nhất với oracle classifier M2 được huấn luyện riêng.
