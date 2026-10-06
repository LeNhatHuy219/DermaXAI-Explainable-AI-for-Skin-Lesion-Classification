# Checkpoints Macro F1

25 weights được chọn theo **diagnosis validation Macro F1 tại ngưỡng 0.5**, hòa giữ epoch đầu tiên. Tên file giữ nguyên sau khi chuyển vào thư mục này.

| Nhóm | Weights |
|---|---:|
| M1 EfficientNet-B0 | 3 |
| M2 Oracle LR | 1 |
| M2 Oracle MLP | 3 |
| M3 Soft CBM: Linear 10 epochs và MLP128 20 epochs | 6 |
| M4-ST: Linear 10 epochs và MLP128 20 epochs | 6 |
| M4-SG: Linear 10 epochs và MLP128 20 epochs | 6 |

[Manifest SHA-256 và JSON nguồn](manifest.json) ghi từng weight. Kết quả nằm trong [results/f1_macro](../../results/README.md#f1-macro). Test/intervention dùng frozen weights và threshold validation; không chọn lại epoch/ngưỡng từ test.

Việc chuyển thư mục không ghi lại checkpoint. Các hash export validation trước test vẫn mô tả artifacts ở thời điểm chúng được tạo.
