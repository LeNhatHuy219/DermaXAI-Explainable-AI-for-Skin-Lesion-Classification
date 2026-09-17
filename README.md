# DermaXAI: Explainable AI for Skin Lesion Classification

> **Khoa luận Tốt nghiệp Đại học**  
> **Đề tài:** Ứng dụng Mô hình Nút thắt Khái niệm (Explainable Concept Bottleneck Models - ECBM) trong Phân loại ảnh có giải thích Tổn thương Da liễu trên Bộ dữ liệu Derm7pt.  

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
| 7 | `vascular_structures` (Cấu trúc mạch máu) | 8 | [20 .. 27] | absent, arborizing, comma, dotted, hairpin, linear irregular, regular, wreath |

* **Biểu diễn nhãn dạng số nguyên (`concept_indices`):** Tensor kích thước `(7,)` lưu trữ index cục bộ của từng concept, phục vụ cho hàm mất mát Multi-Head Cross-Entropy.
* **Biểu diễn nút thắt cổ chai (`concept_onehot`):** Vector 28 chiều ghép từ 7 one-hot vectors, luôn có đúng 7 bit được kích hoạt (tổng vector = 7.0), đưa vào mạng phân loại chẩn đoán $g$.

---

## 3. Kiến trúc Hệ thống Tổng quan (System Architecture)

Quá trình suy luận chẩn đoán được chia tách thành các chặng tuần tự:

| Chặng | Luồng Xử lý | Chi tiết Kỹ thuật | Đầu ra & Ý nghĩa Y khoa |
|:---|:---|:---|:---|
| **1. Đầu vào** | Ảnh soi da (Dermoscopy) | Kích thước 224 x 224 x 3 pixel (LetterboxResize) | Dữ liệu hình ảnh bảo toàn hình học tổn thương |
| **2. Trích xuất** | Mạng Backbone $f$ | Mô hình CNN (ResNet) hoặc ViT | Trích xuất đặc trưng hình ảnh bậc cao |
| **3. Nút thắt** | Bottleneck $c$ | 7 Khái niệm lâm sàng (7-Point Checklist) | Vector 28 ô One-Hot đại diện cho dấu hiệu bệnh lý |
| **4. Suy luận** | Mạng Chẩn đoán $g$ | Bộ phân loại Linear / MLP | Kết quả: 0 (Lành tính) hoặc 1 (Ung thư Melanoma) |
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
* **Nguyên tắc chống rò rỉ dữ liệu (Zero Data Leakage):** Việc phân chia được cố định theo mã định danh bệnh nhân (`case_num`). Một bệnh nhân chỉ xuất hiện duy nhất ở một tập, tuyệt đối không có sự trùng lặp giữa train, valid và test.
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

## 6. Tài liệu Tham khảo Chính (References)

1. Koh, P. W., et al. *"Concept Bottleneck Models."* International Conference on Machine Learning (ICML), 2020.
2. Kawahara, J., et al. *"Seven-Point Checklist and Skin Lesion Classification Using Multitask Multimodal Neural Nets."* IEEE Journal of Biomedical and Health Informatics (JBHI), 2019.
3. Nápoles, G., et al. *"On the Concept Inconsistency in Derm7pt for Explainable Dermatology."* (2026).
4. Patricio, C., et al. *"Coherent Concept-based Explanations for Skin Lesion Analysis."* (2023–2025).
