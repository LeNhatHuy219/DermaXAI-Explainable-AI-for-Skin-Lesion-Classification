# Energy-Based Concept Bottleneck Models: Unifying Prediction, Concept Intervention, and Probabilistic Interpretations

> Source coverage: Full paper
> Extraction confidence: High
> Locator mode: structure-grounded
> Primary analytical lens: methods
> Secondary analytical lens: None
> Context verification: Paper-only
> Card completeness: Complete relative to supplied source

---

## 01 Basic Information

* **Title:** Energy-Based Concept Bottleneck Models: Unifying Prediction, Concept Intervention, and Probabilistic Interpretations [Paper: Title]
* **Authors & Affiliations:** Xinyue Xu (HKUST), Yi Qin (HKUST), Lu Mi (University of Washington), Hao Wang (Rutgers University), Xiaomeng Li (HKUST) [Paper: Authors]
* **Venue & Year:** International Conference on Learning Representations (ICLR) 2024 [Paper: Header]
* **Paper Type:** Methods (Primary)
* **Field:** Interpretable Machine Learning / Concept Bottleneck Models / Energy-Based Models
* **Keywords:** Concept Bottleneck Models (CBM), Energy-Based Models (EBM), Concept Intervention, Probabilistic Interpretation, Dermatology & Image Classification
* **Code Repository:** https://github.com/eeyangqin/ECBM [Paper: Section 4]
* **Benchmark Datasets:** CUB-200-2011, CelebA, Animals with Attributes 2 (AWA2) [Paper: Section 4.1]
* **Position in User's Research:** Bài báo nền tảng cốt lõi (Foundational Paper) cho mô hình đích M5 (Categorical ECBM) trong đề tài khóa luận phân loại tổn thương da trên bộ dữ liệu Derm7pt.

---

## 02 One-Sentence Summary

Bài báo đề xuất mô hình Nút thắt Khái niệm Dựa trên Năng lượng (ECBM) bằng cách biểu diễn sự tương thích giữa ảnh, khái niệm và nhãn lớp qua ba hàm năng lượng kết hợp, từ đó khắc phục hạn chế về tương tác khái niệm, thống nhất quá trình can thiệp và cung cấp khả năng diễn giải xác suất có điều kiện mà không làm suy giảm độ chính xác phân loại [Paper: Abstract].

---

## 03 Research Question

* **Vấn đề cụ thể:** Các mô hình CBM truyền thống (x -> c -> y) giả định các khái niệm độc lập có điều kiện, dẫn đến việc không nắm bắt được tương tác phi tuyến bậc cao giữa các khái niệm và không tính toán được phân phối xác suất có điều kiện tự nhiên giữa khái niệm và nhãn [Paper: Section 1].
* **Tại sao vấn đề quan trọng:** Trong ứng dụng y tế và đời sống, các khái niệm luôn có mối quan hệ phụ thuộc chặt chẽ (ví dụ: các đặc trưng bệnh học cùng xuất hiện). Nếu sửa một khái niệm sai mà không lan truyền sang các khái niệm liên đới, hiệu quả can thiệp của chuyên gia sẽ bị giới hạn nghiêm trọng [Paper: Section 1].
* **Tại sao phương pháp hiện tại chưa đủ:** Các biến thể CBM trước đây (như Sequential CBM, Joint CBM, CEM) hoặc bị mất tương quan giữa các khái niệm, hoặc phải đánh đổi độ chính xác (interpretability tax) khi ép thông tin qua bottleneck hẹp [Paper: Section 1].
* **Câu hỏi nghiên cứu cốt lõi:** *Liệu có thể thống nhất bài toán phân loại, can thiệp lan truyền khái niệm và giải thích xác suất trong một mô hình EBM duy nhất mà không phải đánh đổi độ chính xác hay không?* [Paper: Section 1].

---

## 04 Research Background and Development Path

1. **Standard CBM (Koh et al., 2020):** Đặt nền móng với kiến trúc 2 chặng `x -> c -> y`. Ưu điểm là trực quan, nhưng nhược điểm là hiệu năng giảm sút và các khái niệm bị rời rạc [Paper: Section 2].
2. **Concept Embedding Models - CEM (Zarlenga et al., 2022):** Sử dụng 2 vector nhúng cho mỗi khái niệm để tăng dung lượng biểu diễn, cải thiện độ chính xác nhưng vẫn thiếu cơ chế tính toán xác suất liên kết giữa các khái niệm [Paper: Section 2].
3. **Probabilistic CBM (Kim et al., 2023):** Đưa phân phối xác suất vào khái niệm nhưng chỉ giải quyết được một phần tính bất định, chưa có cơ chế lan truyền can thiệp toàn cục [Paper: Section 2].
4. **ECBM (Xu et al., ICLR 2024 - Bài báo này):** Sử dụng khung lý thuyết Mô hình Dựa trên Năng lượng (Energy-Based Models), mô hình hóa hàm năng lượng đồng thời `E(x, c, y)`, thống nhất suy luận, can thiệp và giải thích [Paper: Section 3].

---

## 05 Core Pain Points Identified by the Paper

| Pain point | Manifestation | Cause or author explanation | Evidence from the paper |
|---|---|---|---|
| Thiếu tương tác giữa các khái niệm | Can thiệp sửa một khái niệm (ví dụ: ngực vàng) không giúp sửa khái niệm tương quan (ví dụ: bụng vàng) | CBM truyền thống dự đoán các khái niệm độc lập từ backbone, không có cầu nối biểu diễn tương tác giữa c_i và c_j | [Paper: Section 1, Figure 2] |
| Không định lượng được phụ thuộc điều kiện | Không trả lời được xác suất xuất hiện khái niệm này khi đã biết khái niệm khác và nhãn lớp | Không mô hình hóa hàm phân phối đồng thời p(x, c, y) | [Paper: Section 1, Section 3.4] |
| Đánh đổi hiệu năng - giải thích (Trade-off) | Mô hình có nút thắt khái niệm thường đạt accuracy thấp hơn mô hình Black-box | Thông tin dự đoán chẩn đoán bị bóp nghẽn hoàn toàn qua vector khái niệm hữu hạn | [Paper: Section 1, Table 1, Table 4] |

---

## 06 Core Idea

1. **Phương pháp bề mặt (Surface method):** Xây dựng ba mạng nơ-ron năng lượng: mạng năng lượng trực tiếp ảnh-lớp `E_class(x, y)`, mạng năng lượng ảnh-khái niệm `E_concept(x, c)`, và mạng năng lượng toàn cục khái niệm-lớp `E_global(c, y)` [Paper: Section 3.1, Figure 1].
2. **Bản chất cốt lõi (Core insight):** Năng lượng thấp đại diện cho tính tương thích cao. Bằng cách định nghĩa năng lượng đồng thời `E_joint(x, c, y)`, việc suy luận nhãn và khái niệm trở thành bài toán tối ưu hóa tìm cực tiểu năng lượng bằng Gradient Descent [Paper: Section 3.2, Algorithm 1]. Khi một khái niệm được can thiệp cố định, năng lượng toàn cục `E_global` tự động truyền gradient để kéo các khái niệm liên đới về trạng thái tương thích nhất [Paper: Section 3.3].
3. **Bài học tổng quát `[Analysis]`:** Không nhất thiết phải ép toàn bộ luồng thông tin đi qua một nút thắt tuần tự duy nhất để có tính giải thích; hoàn toàn có thể duy trì một đường tắt năng lượng trực tiếp kết hợp với ràng buộc toàn cục để vừa đạt độ chính xác tương đương Black-box vừa giữ trọn khả năng can thiệp.

---

## 07 Method Overview

* **Đầu vào (Input):** Ảnh thô `x` [Paper: Section 3.1].
* **Đầu ra (Output):** Vector xác suất khái niệm `b_c` và phân phối xác suất lớp `b_y` [Paper: Section 3.2].
* **Các module chính:**
  * Bộ trích xuất đặc trưng `F(x)` sinh ra vector biểu diễn `z` [Paper: Section 3.1].
  * Mạng năng lượng lớp `G_zu(z, u)` sinh ra `E_class(x, y)` [Paper: Section 3.1, Equation 3].
  * K mạng năng lượng khái niệm `G_zv(z, v_k)` sinh ra `E_concept(x, c_k)` [Paper: Section 3.1, Equation 6].
  * Mạng năng lượng toàn cục `G_vu([v_k], u)` sinh ra `E_global(c, y)` [Paper: Section 3.1, Equation 9].
* **Luồng xử lý (Workflow):**
  * *Huấn luyện:* Ảnh `x`, ground-truth `c`, ground-truth `y` được đưa vào để tối ưu trọng số các mạng năng lượng và các vector nhúng `u`, `v` thông qua hàm mất mát Boltzmann Likelihood [Paper: Section 3.1, Equation 1].
  * *Suy luận:* Đóng băng mạng và embeddings; khởi tạo `e_c`, `e_y`; cập nhật lặp bằng Gradient Descent để giảm `E_joint` cho đến khi hội tụ; chuẩn hóa qua Sigmoid/Softmax [Paper: Section 3.2, Algorithm 1].

---

## 08 Core Module Breakdown

| Module | Function | Why it is needed | Input and output | Supporting evidence | Known or expected effect of removal |
|---|---|---|---|---|---|
| Backbone Extractor F | Trích xuất vector đặc trưng không gian z từ ảnh | Nén ảnh đầu vào thành biểu diễn trừu tượng 1280 chiều | In: x; Out: z | [Paper: Section 3.1] | Không thể đưa ảnh vào các mạng năng lượng |
| Class Energy E_class | Đo độ tương thích trực tiếp giữa đặc trưng z và nhãn y | Bảo toàn khả năng phân loại mạnh như mô hình Black-box | In: (z, u_m); Out: scalar energy | [Paper: Equation 3] | Đo lường thực tế: Bỏ E_class làm giảm nhẹ độ chính xác phân loại [Paper: Table 4] |
| Concept Energy E_concept | Đo độ tương thích giữa đặc trưng z và từng khái niệm c_k | Nhận diện sự hiện diện của từng thuộc tính trên ảnh | In: (z, v_k); Out: scalar energy | [Paper: Equation 6] | Mô hình mất khả năng dự đoán khái niệm từ ảnh |
| Global Energy E_global | Đo mức tương thích giữa toàn bộ vector khái niệm và nhãn y | Nắm bắt tương quan bậc cao giữa các khái niệm và hỗ trợ lan truyền can thiệp | In: ([v_1..v_K], u_m); Out: scalar energy | [Paper: Equation 9] | Đo lường thực tế: Bỏ E_global làm mất khả năng lan truyền can thiệp sang khái niệm liên đới [Paper: Section 3.3] |

---

## 09 Essential Formulas and Symbols

1. **Tổng hàm năng lượng đồng thời:**
   ```text
   E_joint(x, c, y) = E_class(x, y) + lambda_c * E_concept(x, c) + lambda_g * E_global(c, y)
   ```
   * *Ý nghĩa các ký hiệu:* `x` là ảnh; `c` là vector khái niệm; `y` là nhãn lớp; `lambda_c`, `lambda_g` là trọng số cân bằng năng lượng [Paper: Equation 12].
   * *Mục đích:* Làm hàm mục tiêu trong giai đoạn suy luận và can thiệp.

2. **Xác suất có điều kiện Boltzmann cho lớp:**
   ```text
   p(y | x) = exp(-E_class(x, y)) / sum_m exp(-E_class(x, y_m))
   ```
   * *Mục đích:* Chuyển đổi mức năng lượng thành phân phối xác suất phân loại hợp lệ [Paper: Equation 4].

3. **Hàm mất mát huấn luyện toàn phần:**
   ```text
   L_total = L_class(x, y) + lambda_c * L_concept(x, c) + lambda_g * L_global(c, y)
   ```
   * *Mục đích:* Tối ưu đồng thời cả 3 mạng năng lượng và không gian nhúng [Paper: Equation 2].

4. **Xác suất can thiệp và lan truyền (Proposition 3.1):**
   ```text
   p(c_rem, y | x, c_int) = exp(-E_joint(x, c, y)) / sum_{m, c_rem} exp(-E_joint(x, c, y_m))
   ```
   * *Mục đích:* Tính xác suất phân phối đồng thời của các khái niệm còn lại `c_rem` và nhãn `y` sau khi đã can thiệp cố định một phần khái niệm `c_int` [Paper: Equation 13].

---

## 10 Experimental Design and Evidence Chain

* **Tập dữ liệu:** CUB-200-2011 (11,788 ảnh, 112 khái niệm, 200 lớp loài chim), CelebA (202,599 ảnh, 6 khái niệm, 8 lớp khuôn mặt), AWA2 (37,322 ảnh, 85 khái niệm, 50 lớp động vật) [Paper: Section 4.1].
* **Mô hình đối chứng (Baselines):** Standard CBM (Koh et al., 2020), Concept Embedding Models (CEM, Zarlenga et al., 2022), ProbCBM (Kim et al., 2023), Post-hoc CBM (PCBM, Yuksekgonul et al., 2023), Black-box Model [Paper: Section 4.1, Table 1].
* **Thước đo:** Concept Accuracy, Overall Concept Accuracy, Class Accuracy [Paper: Table 1].

| Experiment | Claim tested | Comparison and conditions | Result | Supported conclusion | Unsupported stronger conclusion | Source |
|---|---|---|---|---|---|---|
| Benchmark độ chính xác | ECBM vượt trội CBM và CEM cả về nhận diện khái niệm và nhãn | Chạy 5 seeds ngẫu nhiên trên CUB, CelebA, AWA2 | CUB Class Acc: ECBM 81.2% vs CEM 79.6% vs CBM 75.9% | ECBM dung hòa tốt giữa độ chính xác và tính giải thích | ECBM vượt qua Black-box trên mọi tập dữ liệu (Black-box trên CUB là 82.6%) | [Paper: Table 1, Table 4] |
| Phân tích can thiệp (Intervention) | Can thiệp ECBM giúp sửa các khái niệm liên đới hiệu quả hơn | Tăng tỷ lệ can thiệp khái niệm từ 0% đến 100% | Đường cong chính xác của ECBM luôn nằm trên CEM và CBM ở mọi tỷ lệ | Năng lượng E_global lan truyền sửa lỗi hiệu quả | Can thiệp 1 khái niệm luôn sửa được toàn bộ các khái niệm còn lại | [Paper: Figure 2] |
| Phân tích thành phần (Ablation) | Cả 3 nhánh năng lượng đều đóng góp thiết yếu | Tháo rời từng nhánh năng lượng trên CUB | Bỏ E_class giảm Class Acc từ 81.2% xuống 72.6% | E_class đóng vai trò duy trì độ chính xác cao | Nhánh x-c-y có thể đứng độc lập mà không giảm hiệu năng | [Paper: Table 4] |

---

## 11 Correct Interpretation of the Conclusions

* **Phạm vi tác vụ:** Đã được kiểm chứng trên bài toán phân loại ảnh nhiều lớp với tập thuộc tính nhị phân có sẵn (0/1).
* **Phụ thuộc tính toán:** Quá trình suy luận (Inference) đòi hỏi thực hiện tối ưu hóa lặp bằng Gradient Descent (Algorithm 1) thay vì chỉ lan truyền tiến một lượt, dẫn đến thời gian suy luận trên mỗi mẫu ảnh lâu hơn CBM chuẩn.
* **Biên giới ứng dụng:** Bài báo giả định toàn bộ khái niệm là nhị phân (Binary concepts). Khi áp dụng vào bài toán có khái niệm đa trạng thái (Categorical concepts như Derm7pt), cần mở rộng không gian embedding cho từng trạng thái riêng biệt.
* **Kết luận hợp lệ:** ECBM là mô hình CBM đầu tiên thống nhất thành công suy luận, can thiệp lan truyền và giải thích xác suất trên cơ sở mô hình năng lượng với hiệu năng vượt trội các CBM trước đó [Paper: Section 5].

---

## 12 Limitations Explicitly Acknowledged by the Authors

| Limitation | Specific manifestation | Future direction proposed by the authors | Source |
|---|---|---|---|
| Chi phí tính toán tổ hợp trong E_global | Số lượng tổ hợp khái niệm trong mẫu số của L_global tăng theo hàm mũ 2^K | Sử dụng chiến lược lấy mẫu âm (Negative Sampling) để xấp xỉ mẫu số | [Paper: Section 3.1, Equation 10] |
| Chưa mở rộng sang bài toán không nhãn khái niệm | Mô hình hiện tại cần chú thích khái niệm đầy đủ khi huấn luyện | Mở rộng sang bài toán Unsupervised hoặc Label-free concept learning trong tương lai | [Paper: Section 2] |

---

## 13 Critical Analysis

| `[Analysis]` Observation | Potential issue or alternative explanation | Why it matters | How to test it | Basis |
|---|---|---|---|---|
| Nhánh E_class học tắt (Shortcut learning) | Nhánh E_class đi trực tiếp từ z sang y có thể lấn át luồng khái niệm, gây hiện tượng thông tin bệnh học bị rò rỉ ngoài bottleneck | Tính giải thích của c có thể bị giảm vai trò nếu mô hình chỉ dựa vào E_class để phân loại | Thực hiện ablation bỏ E_class hoặc đo mức độ suy giảm khi che mờ đặc trưng ảnh | Quan sát từ bảng Ablation [Paper: Table 4] |
| Suy luận lặp qua Gradient Descent | Tốc độ suy luận phụ thuộc vào số bước lặp hội tụ của Algorithm 1 | Khó triển khai trên các thiết bị y tế biên hoặc hệ thống thời gian thực | Đo độ trễ (latency/FPS) suy luận so sánh giữa M1, M3 và ECBM | Quy trình thuật toán [Paper: Algorithm 1] |

---

## 14 Knowledge Learned

1. **Khái niệm chuyển giao được:** Biểu diễn các thành phần học máy bằng các mức năng lượng tương thích (Energy Compatibility) thay vì chỉ dùng phân loại xác suất một chiều.
2. **Kỹ thuật Embedding cặp:** Biểu diễn mỗi khái niệm bằng 2 vector nhúng đối ngẫu `v(+)` và `v(-)` giúp việc tính toán tương tác trở nên liên tục và khả vi.
3. **Quy trình tối ưu suy luận:** Khởi tạo xác suất mềm và tối ưu trực tiếp bằng Gradient Descent để tìm cấu hình nhãn và khái niệm có năng lượng tối thiểu.

---

## 15 Connections to Existing Knowledge

* **Liên hệ với M1 (Black-box EfficientNet-B0):** M1 trong đề tài của bạn tương đương với trường hợp thu gọn chỉ giữ nhánh `E_class(x, y)` trong Bảng 4 của bài báo.
* **Liên hệ với M3 (Soft Joint CBM):** M3 sử dụng luồng tuần tự `x -> c -> y`. ECBM bổ sung thêm nhánh tương tác toàn cục `E_global(c, y)` và nhánh trực tiếp `E_class(x, y)`.
* **Liên hệ với tập dữ liệu Derm7pt:** Derm7pt có cấu trúc 7 tiêu chuẩn lâm sàng đa trạng thái (28 trạng thái). Việc chuyển đổi từ nhị phân của Xu et al. sang đa lớp chính là trọng tâm của mô hình M5 trong khóa luận.

---

## 16 Research Ideas

### Idea 1: Categorical Energy-Based CBM for Dermatology (Đích đến M5 của đề tài)
* **Originated Limitation:** Bài báo gốc chỉ thiết kế cho khái niệm nhị phân (0 hoặc 1) [Paper: Section 3.1].
* **Core Hypothesis:** Việc mở rộng vector nhúng từ nhị phân sang K_i trạng thái danh mục (Categorical states) cho từng khái niệm trong Derm7pt sẽ giúp ECBM áp dụng hoàn hảo cho tiêu chuẩn 7-point checklist y khoa.
* **Delta from Paper:** Thay thế `v(+)` và `v(-)` bằng ma trận nhúng `V_k` kích thước `(K_i, d)` cho từng trạng thái của 7 khái niệm da liễu; dùng Softmax thay cho Sigmoid trong Algorithm 1.
* **Initial Method:** Xây dựng mô hình M5 trong `src/models.py` với 3 hàm năng lượng được hiệu chỉnh cho nhãn đa lớp và 7 đầu khái niệm đa trạng thái.
* **Validation / How to test:** Chạy benchmark trên phân chia chuẩn Derm7pt (Train 413, Valid 203, Test 395) qua 3 seeds (42, 123, 2026), đối chiếu trực tiếp với M1, M2, M3, M4.
* **Possible Failure Modes:** Bùng nổ không gian trạng thái khi tính `E_global` do 7 khái niệm tạo ra hàng nghìn tổ hợp trạng thái y khoa; cần áp dụng Negative Sampling nghiêm ngặt.
* **Innovation Status:** prior-art checked (Đã đối chiếu với bài báo gốc ICLR 2024 và tài liệu đề cương).
