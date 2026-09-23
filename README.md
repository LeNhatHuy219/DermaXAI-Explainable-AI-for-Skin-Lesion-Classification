# DermaXAI: Explainable AI for Skin Lesion Classification

---

## 1. Mô tả Đề tài và Mục tiêu Nghiên cứu (Problem Description & Objectives)

### 1.1. Bối cảnh và Tính cấp thiết

Ung thư hắc tố (Melanoma) là một trong những dạng ung thư da ác tính và có tỷ lệ tử vong cao nhất nếu không được phát hiện và can thiệp ở giai đoạn sớm. Trong những năm gần đây, các mô hình học sâu (Deep Learning) đạt được độ chính xác rất cao trong phân loại ảnh soi da (dermoscopy). Tuy nhiên, phần lớn các mô hình này hoạt động theo cơ chế **"hộp đen" (Black-box)**: dự đoán trực tiếp từ ảnh sang nhãn bệnh ($x \to y$) mà không thể đưa ra bất kỳ cơ sở y lý hay giải thích nào cho kết quả chẩn đoán.

Sự thiếu minh bạch này dẫn đến nguy cơ:
* Bác sĩ không thể kiểm tra xem mô hình dựa vào dấu hiệu bệnh lý thực sự hay dựa vào các đặc trưng gây nhiễu (như lông, thước đo, bọt gel, viền đen của máy chụp).
* Thiếu tính giải trình và khó được cấp phép áp dụng trong môi trường khám chữa bệnh thực tế.

### 1.2. Mục tiêu Nghiên cứu của Đề tài

Đề tài **DermaXAI** hướng tới xây dựng một hệ thống phân loại tổn thương da vừa đạt hiệu năng chẩn đoán cao, vừa có khả năng giải thích minh bạch theo chuẩn y khoa thông qua mô hình **Concept Bottleneck Models (CBM / ECBM)**:
1. **Minh bạch hóa quá trình suy luận:** Chia tách bài toán thành hai chặng rõ ràng:
   * Chặng 1 ($f: x \to c$): Dự đoán các đặc trưng lâm sàng trung gian dựa trên bảng kiểm 7 điểm chuẩn da liễu (**7-Point Checklist**).
   * Chặng 2 ($g: c \to y$): Đưa ra kết luận chẩn đoán bệnh (**Melanoma** vs. **Non-Melanoma**) hoàn toàn từ các đặc trưng trung gian này.
2. **Khả năng Can thiệp của Bác sĩ (Concept Intervention):** Cho phép bác sĩ chỉnh sửa trực tiếp các đặc trưng bị mô hình đoán sai ở tầng giữa ($c \to c^*$), từ đó định hướng mô hình sửa đổi kết luận chẩn đoán cuối cùng một cách nhất quán.
3. **Nghiên cứu tính bất nhất của khái niệm (Concept Inconsistency):** Phân tích và đề xuất giải pháp cho hiện tượng các mẫu có cùng hồ sơ thuộc tính y khoa nhưng mang nhãn chẩn đoán xung đột trong dữ liệu thực tế.

---

## 2. Cơ sở Y khoa: Bảng kiểm 7 Điểm (7-Point Checklist)

Hệ thống sử dụng trọn vẹn 7 tiêu chuẩn lâm sàng của bảng kiểm Derm7pt, tương ứng với **28 trạng thái bệnh lý rời rạc**:

| STT | Khái niệm lâm sàng (`concept_name`) | Số trạng thái | Vị trí One-Hot | Các trạng thái bệnh lý cụ thể |
|:---:|:---|:---:|:---:|:---|
| 1 | `pigment_network` (Mạng lưới sắc tố) | 3 | [0 .. 2] | absent, typical, atypical |
| 2 | `streaks` (Dải vệt sắc tố) | 3 | [3 .. 5] | absent, regular, irregular |
| 3 | `pigmentation` (Vùng tăng sắc tố) | 5 | [6 .. 10] | absent, diffuse regular, diffuse irregular, localized regular, localized irregular |
| 4 | `regression_structures` (Cấu trúc thoái triển) | 4 | [11 .. 14] | absent, white areas, blue areas, combinations |
| 5 | `dots_and_globules` (Chấm và hạt cầu) | 3 | [15 .. 17] | absent, regular, irregular |
| 6 | `blue_whitish_veil` (Màn mờ xanh trắng) | 2 | [18 .. 19] | absent, present |
| 7 | `vascular_structures` (Cấu trúc mạch máu) | 8 | [20 .. 27] | absent, arborizing, comma, dotted, hairpin, linear irregular, within regression, wreath |

* **Biểu diễn nhãn dạng số nguyên (`concept_indices`):** Tensor kích thước `(7,)` lưu trữ index cục bộ của từng concept, phục vụ cho hàm mất mát Multi-Head Cross-Entropy.
* **Biểu diễn nút thắt cổ chai (`concept_onehot`):** Vector 28 chiều ghép từ 7 one-hot vectors, luôn có đúng 7 bit được kích hoạt (tổng vector = 7.0), đưa vào mạng phân loại chẩn đoán $g$.

---

## 3. Kiến trúc Hệ thống Tổng quan (System Architecture)

Quá trình suy luận chẩn đoán được chia tách thành các chặng tuần tự:

| Chặng | Luồng Xử lý | Chi tiết Kỹ thuật | Đầu ra & Ý nghĩa Y khoa |
|:---|:---|:---|:---|
| **1. Đầu vào** | Ảnh soi da (Dermoscopy) | Kích thước 224 x 224 x 3 pixel (LetterboxResize) | Dữ liệu hình ảnh bảo toàn hình học tổn thương |
| **2. Trích xuất** | Mạng Backbone $f$ | Mô hình CNN (EfficientNet-B0) | Trích xuất đặc trưng hình ảnh bậc cao |
| **3. Nút thắt** | Bottleneck $c$ | 7 Khái niệm lâm sàng (7-Point Checklist) | Vector 28 ô One-Hot đại diện cho dấu hiệu bệnh lý |
| **4. Suy luận** | Mạng Chẩn đoán $g$ | Bộ phân loại Linear / MLP | Kết quả: 0 (Non-Melanoma) hoặc 1 (Melanoma); Non-Melanoma không đồng nghĩa lành tính |
| **5. Can thiệp** | Bác sĩ tương tác ($c^*$) | Chỉnh sửa các đặc trưng bị AI nhận định sai | Cập nhật lại chẩn đoán $y^*$ theo đúng y lệnh |

> **Sơ đồ Kiến trúc Tương tác Chi tiết:**  
> Xem toàn bộ kiến trúc hệ thống trực quan tại file **[architecture.html](architecture.html)** (có thể mở trực tiếp bằng trình duyệt web, hỗ trợ Dark/Light mode, phóng to thu nhỏ và tra cứu mã nguồn từng thành phần).

---

## 4. Dữ liệu và Phân chia Tập Dữ liệu (Dataset & Splits)

Nghiên cứu sử dụng bộ dữ liệu lâm sàng chuẩn quốc tế **Derm7pt**:
* Tổng số ca bệnh: **1.011 ca**.
* **Phân chia tập (Splits):**
  * `train`: 413 mẫu (323 Non-Melanoma, 90 Melanoma - tỷ lệ mất cân bằng ~ 3.6 : 1).
  * `valid`: 203 mẫu.
  * `test`: 395 mẫu.
* **Phân chia cố định theo ca (Zero Leakage):** Việc phân chia được cố định theo mã ca tổn thương (`case_num`), không có sự trùng lặp giữa train, valid và test.
* **Xử lý mất cân bằng lớp:** Trọng số phạt nghịch đảo chỉ được tính toán trên tập train:
  * Lớp 0 (Non-Melanoma): $w_0 = 0.6393$
  * Lớp 1 (Melanoma): $w_1 = 2.2944$ (phạt nặng gấp 3.59 lần khi đoán sai ca ung thư).

---

## 5. Phát hiện Thực nghiệm Hiện tại (Current Findings)

Từ quá trình tiền xử lý và chạy bài audit 80 tiêu chí trên toàn bộ 1.011 ca bệnh:
1. **Hiện tượng Bất nhất Khái niệm (30.3% mẫu):**
   * Có **306 / 1.011 mẫu** có cùng bộ 7 đặc trưng lâm sàng giống hệt nhau nhưng lại có kết luận chẩn đoán khác nhau.
   * **Ý nghĩa khoa học:** Điều này chứng minh rằng việc chẩn đoán chỉ dựa trên 7 khái niệm thuần túy (Pure CBM) sẽ bị chặn trần độ chính xác bởi sự mơ hồ tự nhiên trong dữ liệu y khoa. Đây là luận điểm then chốt để khóa luận đề xuất nhánh **Residual / Hybrid ECBM**.
2. **Trạng thái khái niệm hiếm (Rare States):**
   * Một số trạng thái có rất ít mẫu trong tập train (ví dụ `vascular_structures = hairpin` chỉ có 4 mẫu, `pigmentation = localized regular` có 0 mẫu train và 3 mẫu test). Khi báo cáo F1-Score từng concept trong khóa luận, chỉ số sẽ luôn đi kèm cột độ hỗ trợ (`support`) để đảm bảo tính khách quan.

---

## 6. Chạy M1 EfficientNet-B0

```bash
# Giai đoạn 1: Huấn luyện và chọn ngưỡng trên Validation
python3 experiments/run_m1.py --mode train --seed 42

# Giai đoạn 2: Đánh giá chính thức trên Test Set
python3 experiments/run_m1.py --mode test --seed 42
```

Chế độ `train` chỉ tạo loader train/validation và không đánh giá test. Checkpoint được chọn theo validation Balanced Accuracy tại ngưỡng 0.5; ngưỡng quyết định cuối được chọn trên validation của checkpoint đó bằng cách tối đa Balanced Accuracy. Chế độ `test` sử dụng checkpoint và ngưỡng đã đóng băng để đánh giá một lần duy nhất trên tập test độc lập.

---

## 7. Chạy M2 - Oracle Concept Model

**Mục đích:** Đo trần thông tin (information ceiling) mà 7 concept Derm7pt Ground Truth thực sự chứa được cho bài toán chẩn đoán, hoàn toàn không dùng ảnh. Bộ phân loại $g$ là Logistic Regression (Linear, `class_weight="balanced"`) trên vector concept one-hot 28 chiều `c → y`. Hệ số C được chọn bằng grid-search tối đa Balanced Accuracy trên validation (tại ngưỡng 0.5); ngưỡng quyết định cuối cùng sau đó được chọn trên validation của mô hình đã chọn C, theo đúng quy trình 2 bước (chọn hyperparameter, rồi chọn ngưỡng) như M1 để tránh rò rỉ thông tin từ test.

```bash
# Giai đoạn 1: Grid-search C và chọn ngưỡng trên Validation
python3 experiments/run_m2.py --mode train --seed 42

# Giai đoạn 2: Đánh giá chính thức trên Test Set
python3 experiments/run_m2.py --mode test --seed 42
```

Kết quả M2 cũng lưu lại hệ số hồi quy (`concept_state_coefficients`) cho từng trạng thái trong 28 chiều one-hot, phục vụ phân tích trạng thái nào đóng góp mạnh nhất vào quyết định Melanoma/Non-Melanoma.

> **Lưu ý môi trường (Windows CPU-only):** import `scikit-learn` **trước** khi import `torch`/`torchvision` trong cùng một process có thể gây `Segmentation fault` do xung đột thứ tự nạp DLL OpenMP/MKL trên một số máy Windows. Cả `run_m2.py` và `run_m3.py` đều cố tình import `src.dataset` (kéo theo torch) trước `sklearn` để tránh lỗi này — giữ nguyên thứ tự import ở đầu hai file này nếu chỉnh sửa.

---

## 8. Chạy M3 - Soft Joint CBM

**Mục đích:** CBM baseline chính để so sánh với ECBM (M5). Kiến trúc: EfficientNet-B0 (giống backbone của M1) trích đặc trưng 1280 chiều → 7 concept head Linear độc lập dự đoán xác suất mềm (softmax) từng nhóm → nối lại thành vector bottleneck 28 chiều (mỗi nhóm con tổng = 1) → đầu chẩn đoán $g$ là **Linear thuần 28→2** (không hidden layer), giống hệt kiến trúc của M2 để hai mô hình so sánh công bằng: sự khác biệt hiệu năng M2 (concept Ground Truth) so với M3 (concept dự đoán từ ảnh) phản ánh đúng phần thông tin bị mất khi phải dự đoán concept từ ảnh thay vì đọc trực tiếp từ nhãn.

Huấn luyện **joint** (đồng thời) toàn bộ backbone + concept heads + $g$ bằng một hàm mất mát tổng hợp:

$$L = L_{\text{diagnosis}}(\text{Weighted CE}) + \lambda \cdot \frac{1}{7}\sum_{i=1}^{7} L_{\text{concept}_i}(\text{CE})$$

Gradient của $L_{\text{diagnosis}}$ truyền ngược xuyên qua vector concept soft (differentiable) tới tận backbone — đây là điểm khác biệt then chốt so với CBM "sequential/independent" (huấn luyện $f$ và $g$ tách rời).

```bash
# Giai đoạn 1: Huấn luyện joint và chọn ngưỡng trên Validation (lambda mặc định = 1.0)
python3 experiments/run_m3.py --mode train --seed 42

# Giai đoạn 2: Đánh giá chính thức trên Test Set
python3 experiments/run_m3.py --mode test --seed 42
```

Cũng như M1, chế độ `train` chỉ dùng train/validation; checkpoint chọn theo validation Balanced Accuracy chẩn đoán tại ngưỡng 0.5, ngưỡng cuối cùng chọn trên validation của checkpoint đó. Kết quả lưu thêm `validation_concept_metrics`/`test_concept_metrics` (Macro-F1 từng concept, cả "all-defined" và "train-observed") để phân tích riêng độ chính xác của tầng bottleneck $f: x \to c$, tách biệt khỏi độ chính xác chẩn đoán cuối $g: c \to y$.

Có thể chỉnh trọng số $\lambda$ giữa concept loss và diagnosis loss bằng `--concept_loss_weight` (mặc định `1.0`).

---

## 9. So sánh Kết quả M0 - M3 trên Test Set (395 mẫu)

Kết quả thực tế đã chạy (xem file JSON tương ứng trong `results/`):

| Model | Balanced Acc | Sensitivity | Specificity | F1 (Melanoma) | ROC-AUC | Ghi chú |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **M0** Majority Baseline | 50.00% | 0.00% | 100.00% | 0.0000 | 0.50 | Mốc sàn, luôn đoán Non-Melanoma |
| **M1** Black-box EfficientNet-B0 | 72.68% | 62.38% | 82.99% | 0.5888 | 0.8356 | Không dùng concept |
| **M2** Oracle Concept (LR trên GT) | **85.37%** | 89.11% | 81.63% | 0.7347 | 0.9211 | Trần thông tin của 7 concept Ground Truth |
| **M3** Soft Joint CBM | 70.64% | 62.38% | 78.91% | 0.5575 | 0.7565 | $g$ Linear giống M2, nhưng $c$ dự đoán từ ảnh (Concept Acc trung bình chỉ ~70%) |

**Nhận xét chính:**
* **M2 (85.37%) >> M1 (72.68%) > M3 (70.64%)**: khoảng cách rất lớn giữa M2 và M3 (~15 điểm Balanced Accuracy) cho thấy tầng bottleneck $f: x \to c$ hiện tại dự đoán concept chưa đủ tốt (Concept Macro-F1 trung bình trên test chỉ **0.4071**, riêng `vascular_structures` chỉ 0.1227 do các trạng thái hiếm) — đây chính là phần hiệu năng "mất đi" khi ép mô hình phải suy luận qua concept thay vì học trực tiếp từ ảnh như M1.
* M3 hiện *thấp hơn* M1 một chút vì cùng một backbone EfficientNet-B0 nhưng M3 phải "chia sẻ" khả năng biểu diễn của backbone cho 7 concept head thay vì tối ưu hoàn toàn cho mục tiêu chẩn đoán, cộng với nhiễu lan truyền từ concept dự đoán sai vào $g$.
* M2 chứng minh rằng nếu tầng concept được dự đoán chính xác (bằng Ground Truth), $g$ Linear đơn giản vẫn đạt hiệu năng cao hơn cả M1 — đây là động lực chính để đề tài đề xuất **M5 Categorical ECBM**: cải thiện tầng $f$ (qua năng lượng $E_{\text{concept}}$) và mô hình hóa tương tác Ảnh↔Concept↔Diagnosis chặt chẽ hơn CBM Joint thông thường, đồng thời hỗ trợ concept intervention để bác sĩ có thể "kéo" M3 tiến gần hơn tới trần M2.
* Khoảng cách M2-M3 cũng là bằng chứng định lượng bổ sung cho hiện tượng Concept Inconsistency (mục 5): dù M2 dùng đúng Ground Truth, nó không đạt 100% vì 30.3% mẫu có hồ sơ concept giống nhau nhưng nhãn chẩn đoán khác nhau — đây là **trần lý thuyết tuyệt đối** của mọi CBM thuần túy trên Derm7pt, và M2 (85.37%) đã tiến rất gần trần đó.

---

## 10. Tài liệu Tham khảo Chính (References)

1. Koh, P. W., et al. *"Concept Bottleneck Models."* International Conference on Machine Learning (ICML), 2020.
2. Kawahara, J., et al. *"Seven-Point Checklist and Skin Lesion Classification Using Multitask Multimodal Neural Nets."* IEEE Journal of Biomedical and Health Informatics (JBHI), 2019.
3. Nápoles, G., Grau, I. & Salgueiro, Y. *"Concept inconsistency in dermoscopic concept bottleneck models: a rough-set analysis of the Derm7pt dataset."* Scientific Reports (2026).
4. Patricio, C., et al. *"Coherent Concept-based Explanations for Skin Lesion Analysis."* (2023–2025).
