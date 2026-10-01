# DermaXAI: Explainable AI for Skin Lesion Classification

## Trạng thái triển khai

Repo hiện có code và kết quả **M0–M3**. **M4 Hard Joint CBM** và **M5 categorical ECBM** là kế hoạch nghiên cứu, chưa có code hoặc kết quả thực nghiệm.

| Mô hình | Trạng thái | Runner |
|---|---|---|
| M0 Majority baseline | Đã hoàn thành | `experiments/run_m0.py` |
| M1 EfficientNet-B0 | Đã hoàn thành, 3 seeds | `experiments/run_m1.py` |
| M2 Oracle LR / MLP | Đã hoàn thành; MLP có 3 seeds | `experiments/run_m2_lr.py`, `experiments/run_m2_mlp.py` |
| M3 Soft Joint CBM | Đã hoàn thành, 3 seeds | `experiments/run_m3.py` |
| M4 Hard Joint CBM | Dự kiến dùng concept dự đoán one-hot thay cho xác suất mềm | Chưa triển khai |
| M5 Categorical ECBM | Dự kiến phân tích các thành phần năng lượng và nhóm concept bất nhất | Chưa triển khai |

ECBM đã được dùng làm đối chứng trên Derm7pt trong [Wang et al., MIDL 2026](https://proceedings.mlr.press/v315/wang26a.html). Kế hoạch của dự án là đánh giá sâu hơn categorical ECBM trên official split; phần này chưa có kết quả để kết luận về hiệu quả hoặc tính mới.

## Cài đặt và chạy từ máy khác

Chạy các lệnh dưới đây trong thư mục repo. Bộ test hiện được kiểm tra với Python 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Trên Windows, kích hoạt môi trường bằng `.venv\Scripts\Activate.ps1` trong PowerShell.

**Dữ liệu được tải riêng** từ [trang chính thức Derm7pt](https://derm.cs.sfu.ca/). Giải nén để có `Derm7pt/release_v0/images/` và `Derm7pt/release_v0/meta/` bên trong thư mục repo. `data/manifest.csv` và `data/label_mapping.json` đã có sẵn; đường dẫn ảnh trong manifest là đường dẫn tương đối nên có thể dùng trên máy khác.

README đi kèm Derm7pt ghi rõ ảnh không được phân phối lại. Thư mục dữ liệu, checkpoint và log cục bộ nằm trong `.gitignore`. Output notebook chứa ảnh mẫu Derm7pt cũng cần được giữ ở máy cục bộ.

Xem kết quả M0–M3 đã lưu ở bảng trong mục 9. Để chạy lại và giữ nguyên các kết quả đã công bố, ghi JSON mới vào `results/local/`:

```bash
python experiments/run_m2_lr.py --checkpoint_path checkpoints/local/m2_lr_best.joblib --results_path results/local/m2_lr_results.json
python experiments/run_m1.py --seed 42 --checkpoint_path checkpoints/local/m1_seed42_best.pth --results_path results/local/m1_seed42_results.json
python experiments/run_m2_mlp.py --seed 42 --checkpoint_path checkpoints/local/m2_mlp_seed42_best.pth --results_path results/local/m2_mlp_seed42_results.json
python experiments/run_m3.py --seed 42 --checkpoint_path checkpoints/local/m3_seed42_best.pth --results_path results/local/m3_seed42_results.json
```

Các lệnh trên tải pretrained ImageNet weights khi cần và chạy train → validation selection → test. Checkpoint được tải về hoặc tạo tại máy người chạy; repo không kèm trọng số. Lệnh intervention cần checkpoint M3 và test export tương ứng. Các đường dẫn tuyệt đối trong JSON cũ ghi lại nơi chạy thí nghiệm ban đầu, không phải đường dẫn cần tạo trên máy mới.

Kiểm tra code và dữ liệu đã tải bằng:

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B tests/test_data_pipeline.py
python -B tests/audit_pipeline.py
```

---

## 1. Mô tả Đề tài và Mục tiêu Nghiên cứu (Problem Description & Objectives)

### 1.1. Bối cảnh và Tính cấp thiết

Ung thư hắc tố (Melanoma) là một trong những dạng ung thư da ác tính và có tỷ lệ tử vong cao nhất nếu không được phát hiện và can thiệp ở giai đoạn sớm. Trong những năm gần đây, các mô hình học sâu (Deep Learning) đạt được độ chính xác rất cao trong phân loại ảnh soi da (dermoscopy). Tuy nhiên, phần lớn các mô hình này hoạt động theo cơ chế **"hộp đen" (Black-box)**: dự đoán trực tiếp từ ảnh sang nhãn bệnh ($x \to y$) mà không thể đưa ra bất kỳ cơ sở y lý hay giải thích nào cho kết quả chẩn đoán.

Sự thiếu minh bạch này dẫn đến nguy cơ:
* Bác sĩ không thể kiểm tra xem mô hình dựa vào dấu hiệu bệnh lý thực sự hay dựa vào các đặc trưng gây nhiễu (như lông, thước đo, bọt gel, viền đen của máy chụp).
* Thiếu tính giải trình và khó được cấp phép áp dụng trong môi trường khám chữa bệnh thực tế.

### 1.2. Mục tiêu Nghiên cứu của Đề tài

Đề tài **DermaXAI** hướng tới xây dựng một hệ thống phân loại tổn thương da vừa đạt hiệu năng chẩn đoán cao, vừa có khả năng giải thích minh bạch theo chuẩn y khoa thông qua mô hình **Concept Bottleneck Models (CBM / ECBM)**:
1. **Đánh giá mô hình concept-based:** Với CBM, chia bài toán thành hai chặng:
   * Chặng 1 ($f: x \to c$): Dự đoán các đặc trưng lâm sàng trung gian dựa trên bảng kiểm 7 điểm chuẩn da liễu (**7-Point Checklist**).
   * Chặng 2 ($g: c \to y$): Đưa ra kết luận chẩn đoán bệnh (**Melanoma** vs. **Non-Melanoma**) hoàn toàn từ các đặc trưng trung gian này.
   * ECBM dùng năng lượng chung ảnh–concept–diagnosis và có class energy trực tiếp từ ảnh; concept profile chưa tự giải thích đầy đủ mọi bằng chứng của quyết định.
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

Index của từng state được lấy từ `data/label_mapping.json`; danh sách mô tả trong bảng không định nghĩa thứ tự encoding. Checkpoint mới đóng băng mapping này.

* **Biểu diễn nhãn dạng số nguyên (`concept_indices`):** Tensor kích thước `(7,)` lưu trữ index cục bộ của từng concept, phục vụ cho hàm mất mát Multi-Head Cross-Entropy.
* **Biểu diễn nút thắt cổ chai (`concept_onehot`):** Vector 28 chiều ghép từ 7 one-hot vectors, luôn có đúng 7 bit được kích hoạt (tổng vector = 7.0), đưa vào mạng phân loại chẩn đoán $g$.

---

## 3. Kiến trúc Hệ thống Tổng quan (System Architecture)

Quá trình suy luận chẩn đoán được chia tách thành các chặng tuần tự:

| Chặng | Luồng Xử lý | Chi tiết Kỹ thuật | Đầu ra & Ý nghĩa Y khoa |
|:---|:---|:---|:---|
| **1. Đầu vào** | Ảnh soi da (Dermoscopy) | Kích thước 224 x 224 x 3 pixel (LetterboxResize) | Dữ liệu hình ảnh bảo toàn hình học tổn thương |
| **2. Trích xuất** | Mạng Backbone $f$ | Mô hình CNN (EfficientNet-B0) | Trích xuất đặc trưng hình ảnh bậc cao |
| **3. Nút thắt** | Bottleneck $c$ | 7 nhóm concept (7-Point Checklist) | M3: xác suất soft 28 chiều; M4 dự kiến: one-hot |
| **4. Suy luận** | Mạng Chẩn đoán $g$ | Bộ phân loại Linear / MLP | Kết quả: 0 (Non-Melanoma) hoặc 1 (Melanoma); Non-Melanoma không đồng nghĩa lành tính |
| **5. Hiệu chỉnh** | Ground-truth intervention ($c^*$) | Thay group bằng annotation one-hot | Chạy lại diagnosis; kết quả có thể tốt lên hoặc xấu đi |

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
* **Phân chia cố định theo ca:** Splits không trùng `case_num`. Điều này chưa tự chứng minh độc lập ở cấp bệnh nhân hoặc loại trừ ảnh gần giống; báo cáo phạm vi kiểm tra trùng ảnh riêng.
* **Xử lý mất cân bằng lớp:** Trọng số phạt nghịch đảo chỉ được tính toán trên tập train:
  * Lớp 0 (Non-Melanoma): $w_0 = 0.6393$
  * Lớp 1 (Melanoma): $w_1 = 2.2944$ (phạt nặng gấp 3.59 lần khi đoán sai ca ung thư).

---

## 5. Phát hiện Thực nghiệm Hiện tại (Current Findings)

Từ quá trình tiền xử lý và chạy bài audit 80 tiêu chí trên toàn bộ 1.011 ca bệnh:
1. **Hiện tượng Bất nhất Khái niệm (30.3% mẫu):**
   * Có **306 / 1.011 mẫu** thuộc các hồ sơ có cùng bộ nhãn 7 concept nhưng khác nhãn chẩn đoán. Thống kê này đã được công bố bởi [Nápoles et al., Scientific Reports 2026](https://www.nature.com/articles/s41598-026-56927-2) và được tái lập trong dự án.
   * Đây là thống kê tái lập từ Derm7pt; tỷ lệ mẫu thuộc profiles xung đột không đồng nghĩa tỷ lệ annotation sai hoặc tỷ lệ lỗi tối thiểu. Nếu mỗi hồ sơ ground-truth phải ánh xạ sang một diagnosis duy nhất trên toàn bộ dữ liệu, bound accuracy là 931/1.011 (khoảng 92,1%). Bound có điều kiện này không phải trần cho test Balanced Accuracy hoặc model dùng thêm thông tin ảnh.
2. **Trạng thái khái niệm hiếm (Rare States):**
   * Một số trạng thái có rất ít mẫu trong tập train (ví dụ `vascular_structures = hairpin` chỉ có 4 mẫu, `pigmentation = localized regular` có 0 mẫu train và 3 mẫu test). Khi báo cáo F1-Score từng concept trong khóa luận, chỉ số sẽ luôn đi kèm cột độ hỗ trợ (`support`) để đảm bảo tính khách quan.

---

## 6. Chạy M1 EfficientNet-B0

```bash
# Một lệnh: train → chọn checkpoint/ngưỡng bằng validation → test
python3 experiments/run_m1.py --seed 42
```

Mặc định (hoặc `--mode train`), M1, M2 LR, M2 MLP và M3 chạy toàn bộ quy trình bằng một lệnh. Bước huấn luyện chỉ dùng train/validation; checkpoint được chọn theo validation Balanced Accuracy tại ngưỡng 0.5, rồi ngưỡng quyết định được chọn trên validation của checkpoint đó bằng cách tối đa Balanced Accuracy. Sau đó chương trình nạp lại checkpoint và ngưỡng đã đóng băng để đánh giá test; test không tham gia lựa chọn mô hình hoặc ngưỡng.

Checkpoint lưu trong `checkpoints/`; mỗi model/seed có **một JSON** trong `results/`, hậu tố `_results.json`, chứa `validation_metrics`, `validation_predictions`, `test_metrics`, `test_predictions`, cấu hình và thông tin huấn luyện. Ví dụ: `m1_efficientnet_b0_seed42_results.json`, `m2_oracle_lr_results.json`, `m2_oracle_mlp_seed42_results.json`. `--results_path` tùy chỉnh file JSON này; đường dẫn phải khác checkpoint. Chương trình kiểm tra file đã tồn tại trước khi train; dùng `--overwrite` nếu muốn chạy lại. File được lưu sau bước validation, rồi cập nhật thêm test khi hoàn tất. `--mode test --overwrite` cập nhật phần test và giữ phần validation của cùng run đã lưu. `--no_save` (M1, M2 LR, M3) bỏ lưu JSON nhưng vẫn lưu checkpoint.

---

## 7. Chạy M2 - Oracle Concept Model

**Mục đích:** Đánh giá concept sufficiency bằng classifier nhận bảy concept ground-truth, không dùng ảnh. LR (`class_weight="balanced"`) là baseline tuyến tính; oracle MLP nhỏ được báo cáo riêng để khảo sát tương tác phi tuyến. Kết quả phụ thuộc classifier, regularization và khả năng tổng quát hóa, không phải cận trên tuyệt đối. LR chọn C theo validation Balanced Accuracy@0.5, rồi chọn threshold bằng validation của model đã chọn.

```bash
# Grid-search C → chọn ngưỡng bằng validation → tự động test
python3 experiments/run_m2_lr.py
```

M2 LR lưu hệ số hồi quy theo state. Diễn giải hệ số cần xét encoding và regularization; hệ số chưa đo tầm quan trọng tổng quát hoặc quan hệ nhân quả.

```bash
# Oracle MLP: train → validation selection → test cho từng seed
for m2_seed in 42 123 2026; do
  python3 experiments/run_m2_mlp.py --seed "$m2_seed" || break
done
```

MLP mặc định 28→32→2, ReLU, dropout 0.2, AdamW, 100 epochs tối đa. Đây là cấu hình khởi đầu; chốt tuning budget bằng validation trước test. LR lbfgs thường không thay đổi nghiệm khi đổi seed trên dữ liệu cố định; oracle MLP cần nhiều seeds vì khởi tạo và minibatch ngẫu nhiên.

> **Lưu ý môi trường (Windows CPU-only):** import `scikit-learn` **trước** khi import `torch`/`torchvision` trong cùng một process có thể gây `Segmentation fault` do xung đột thứ tự nạp DLL OpenMP/MKL trên một số máy Windows. Cả `run_m2_lr.py` và `run_m3.py` đều cố tình import `src.dataset` (kéo theo torch) trước `sklearn` để tránh lỗi này — giữ nguyên thứ tự import ở đầu hai file này nếu chỉnh sửa.

---

## 8. Chạy M3 - Soft Joint CBM

**Mục đích:** CBM baseline chính để so sánh với ECBM (M5). Kiến trúc: EfficientNet-B0 → bảy concept heads Linear → softmax từng group → vector 28 chiều → diagnosis head Linear 28→2. M2 LR và M3 cùng họ head tuyến tính nhưng khác input representation, objective và cách tối ưu. Chênh lệch kết quả chưa tách riêng ảnh hưởng của concept errors; soft probabilities có thể mang thêm thông tin qua các mức xác suất.

Huấn luyện **joint** (đồng thời) toàn bộ backbone + concept heads + $g$ bằng một hàm mất mát tổng hợp:

$$L = L_{\text{diagnosis}}(\text{Weighted CE}) + \lambda \cdot \frac{1}{7}\sum_{i=1}^{7} L_{\text{concept}_i}(\text{State-weighted CE})$$

Gradient của $L_{\text{diagnosis}}$ truyền ngược xuyên qua vector concept soft (differentiable) tới tận backbone — đây là điểm khác biệt then chốt so với CBM "sequential/independent" (huấn luyện $f$ và $g$ tách rời).

```bash
# Huấn luyện joint → chọn checkpoint/ngưỡng bằng validation → tự động test
python3 experiments/run_m3.py --seed 42
```

Cũng như M1, bước huấn luyện chỉ dùng train/validation; checkpoint chọn theo validation Balanced Accuracy chẩn đoán tại ngưỡng 0.5, ngưỡng cuối cùng chọn trên validation của checkpoint đó, sau đó tự động đánh giá test. Kết quả lưu thêm `validation_concept_metrics`/`test_concept_metrics` (Macro-F1 từng concept, cả "all-defined" và "train-observed") để phân tích riêng độ chính xác của tầng bottleneck $f: x \to c$, tách biệt khỏi độ chính xác chẩn đoán cuối $g: c \to y$.

Có thể chỉnh trọng số $\lambda$ giữa concept loss và diagnosis loss bằng `--concept_loss_weight` (mặc định `1.0`).

M3 được chốt là **một Soft Joint CBM với concept-state weighted CE**. State weights chỉ tính từ train, $w_s=N/(K_{observed}N_s)$; state không có mẫu train nhận weight 0. Batch đánh giá chỉ gồm states weight 0 có concept loss 0 và vẫn được tính trong metrics all-defined. CLI chỉ có một formulation; config lưu `protocol=soft_joint_state_weighted_v1` để xác nhận checkpoint thuộc pipeline đã chốt.

Export mới lưu GT/prediction/probabilities của từng concept theo case ID, exact-match, per-state F1/support và confusion matrix. Checkpoint M2/M3 lưu concept schema gồm mapping, order, cardinalities, offsets và hash; mapping phải nằm cạnh manifest. Hash manifest chuẩn hóa LF/CRLF nhưng vẫn kiểm tra thay đổi nội dung. M2/M3 được train lại bằng pipeline mới để tạo đầy đủ metadata. M3 test/intervention yêu cầu checkpoint đúng formulation đã chốt.

### Cấu hình và lệnh train M3 đã chốt

Giữ cùng cấu hình cho cả ba seeds:

| Tham số | Giá trị |
|---|---|
| Backbone | EfficientNet-B0 pretrained ImageNet |
| Diagnosis head | Linear 28→2 |
| Dropout | 0.2 |
| Loss | Weighted diagnosis CE + 1.0 × mean của 7 state-weighted concept CEs |
| Epochs / batch size | 10 / 16 |
| Optimizer / learning rate / weight decay | AdamW / 1e-4 / 1e-2 |
| Scheduler | CosineAnnealingLR, eta_min=1e-6 |
| Augmentation | legacy_letterbox |
| Seeds | 42, 123, 2026 |

```bash
for m3_seed in 42 123 2026; do
  python3 experiments/run_m3.py --seed "$m3_seed" --num_workers 0 || break
done
```

Các lệnh dùng defaults trong bảng và chạy từ thư mục repo, trong môi trường đã cài dependencies. Device tự chọn CUDA, MPS hoặc CPU; có thể chỉ định `--device`. Tên artifact mặc định gồm model và seed, ví dụ `m3_soft_joint_cbm_seed42_best.pth`. Chạy mới tạo artifact riêng cho mỗi seed.

### Intervention M3

Chạy đủ ba seeds và xuất bảng/biểu đồ bằng một lệnh, dùng checkpoint và test predictions hiện có:

```bash
python3 experiments/run_m3_interventions.py
```

Kết quả nằm trong [`results/m3_intervention/report.md`](results/m3_intervention/report.md): JSON chi tiết của mỗi seed, `summary.json`, `curve.csv`, `per_concept.csv`, `full_intervention_cases.csv`, biểu đồ PNG và SVG. Lệnh không train lại. Mặc định từ chối ghi đè; dùng `--output_dir` để tạo bộ kết quả riêng hoặc `--overwrite` để cập nhật bộ đã có. Chạy trong môi trường có dependencies của dự án.

Đã hoàn thành 128 subsets × 395 ca × ba seeds 42/123/2026. Baseline m=0 được tái hiện từ frozen head và khớp test export. Balanced Accuracy trung bình giảm từ **70.30% ± 4.58 điểm %** ở m=0 xuống **53.19% ± 2.83 điểm %** khi sửa đủ bảy groups; Macro-F1 giảm từ **0.6844 ± 0.0326** xuống **0.5262 ± 0.0243**. Đây là kết quả mô tả trên checkpoints hiện tại, chưa phải kết luận về nguyên nhân. Soft probabilities → GT one-hot thay đổi representation đầu vào của diagnosis head; ảnh hưởng của thay đổi này cần được khảo sát riêng. Kết quả chưa chứng minh concept leakage, inconsistency là nguyên nhân hoặc ECBM tốt hơn. M2 oracle được huấn luyện riêng nên không đồng nhất với M3 sau full intervention.

Nếu chỉ cần chạy một seed:

```bash
python3 experiments/run_m3.py --mode intervention --seed 42
python3 experiments/run_m3.py --mode intervention --seed 123
python3 experiments/run_m3.py --mode intervention --seed 2026
```

Train, test và intervention dùng chung `run_m3.py`. Intervention tự tìm checkpoint/export test theo seed và config; nếu dùng đường dẫn riêng thì truyền `--checkpoint_path` và `--predictions_path`. Evaluator đọc export test mới và head đã đóng băng, xác nhận hash checkpoint, config, schema, labels, case IDs và tái hiện baseline diagnosis. Duyệt đủ 128 subsets; thay cả group bằng GT one-hot, giữ groups chưa sửa và threshold cố định. Metrics được tính từng subset trước khi lấy trung bình theo m/7. Xuất diagnosis trước/sau, delta xác suất, ca được sửa đúng/xấu đi, độ đúng concepts chưa sửa (m=0…6), cùng intervention-curve AUC. SD giữa exhaustive subsets được ghi riêng với SD giữa seeds và CI theo mẫu. Đây là mô phỏng hiệu chỉnh annotation; tác động hiệu chỉnh chưa đo tầm quan trọng tổng quát hay quan hệ nhân quả.

---

## 9. So sánh Kết quả M0 - M3 trên Test Set (395 mẫu)

Kết quả hiện tại lấy từ các JSON trong `results/`. M1, M2 MLP và M3 dùng ba seeds 42, 123, 2026, báo cáo mean ± sample SD. M0 là baseline xác định; M2 LR dùng một seed với solver lbfgs. SD của Balanced Accuracy có đơn vị điểm phần trăm. Hai exports M3 unweighted cũ đã được xoá sau khi hoàn thành bộ M3 weighted.

| Model | Balanced Acc | Macro-F1 | ROC-AUC | AP | Runs |
|:---|:---:|:---:|:---:|:---:|:---:|
| **M0** Majority Baseline | 50.00% | 0.4267 | 0.5000 | 0.2557 | 1 |
| **M1** Black-box EfficientNet-B0 | 70.74% ± 1.88 | 0.7078 ± 0.0118 | 0.8302 ± 0.0062 | 0.6608 ± 0.0115 | 3 seeds |
| **M2 LR** Oracle trên GT concepts | 85.37% | 0.8077 | 0.9211 | 0.7848 | 1 seed |
| **M2 MLP** Oracle trên GT concepts | 86.23% ± 1.27 | 0.8220 ± 0.0154 | 0.9246 ± 0.0029 | 0.7981 ± 0.0043 | 3 seeds |
| **M3** Weighted Soft Joint CBM | 70.30% ± 4.58 | 0.6844 ± 0.0326 | 0.7616 ± 0.0446 | 0.5917 ± 0.0240 | 3 seeds |

**Nhận xét mô tả:** Hai oracle M2 nhận GT concepts và đạt Balanced Accuracy cao hơn các models dùng ảnh trong bộ runs này. M2 MLP cao hơn LR khoảng 0.86 điểm phần trăm; M1 cao hơn M3 weighted khoảng 0.44 điểm phần trăm. Chưa thực hiện paired bootstrap để kết luận về chênh lệch. Khoảng cách M2–M3 chưa tự xác lập nguyên nhân hoặc chứng minh inconsistency. Lợi ích ECBM được kết luận sau thực nghiệm, kể cả trường hợp không cải thiện.

### M1: ba seeds đã hoàn thành

M1 đã hoàn thành train/validation và test cho seeds 42, 123, 2026 với cấu hình 10 epochs, batch 16, LR 1e-4, weight decay 1e-2, legacy_letterbox. Checkpoint tốt nhất lần lượt ở epochs 7, 7, 8; threshold lấy từ validation của từng checkpoint. Các metrics test đã được tính lại từ prediction records và tổng hợp trong [m1_three_seeds_summary.json](results/m1_three_seeds_summary.json).

| Seed | Test Balanced Accuracy | Test Macro-F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|
| 42 | 72.68% | 0.7180 | 0.8356 | 0.6483 |
| 123 | 68.94% | 0.6949 | 0.8234 | 0.6633 |
| 2026 | 70.60% | 0.7106 | 0.8317 | 0.6707 |
| Mean ± sample SD | 70.74% ± 1.88 điểm % | 0.7078 ± 0.0118 | 0.8302 ± 0.0062 | 0.6608 ± 0.0115 |

Đây là baseline M1 qua ba seeds; dùng đầy đủ bộ runs để so sánh với M3 đã chốt cùng training budget. SD giữa seeds không thay thế paired bootstrap theo case IDs cho chênh lệch quan trọng.

### M3 weighted: ba seeds đã hoàn thành

M3 đã train và test cho seeds **42, 123, 2026** trên MPS, cùng cấu hình 10 epochs, batch 16, LR 1e-4, weight decay 1e-2, lambda 1.0, legacy_letterbox và 2 DataLoader workers. Mỗi lệnh tự train → chọn checkpoint/threshold bằng validation → test. Checkpoint tốt nhất lần lượt ở epochs **8, 6, 5**; thresholds tương ứng **0.455935, 0.443535, 0.444767**.

| Seed | Test Balanced Accuracy | Test Macro-F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|
| 42 | 66.20% | 0.6561 | 0.7126 | 0.5677 |
| 123 | 69.47% | 0.6772 | 0.7723 | 0.6157 |
| 2026 | 75.24% | 0.7200 | 0.7999 | 0.5916 |
| Mean ± sample SD | 70.30% ± 4.58 điểm % | 0.6844 ± 0.0326 | 0.7616 ± 0.0446 | 0.5917 ± 0.0240 |

Concept accuracy trung bình là **59.61% ± 2.50 điểm %**, mean concept Macro-F1 all-defined **0.4570 ± 0.0185**, exact match cả bảy concepts **4.22% ± 0.81 điểm %**. Kết quả lưu trong [m3_three_seeds_summary.json](results/m3_three_seeds_summary.json); log từng seed nằm trong `results/logs/`. Đã xác nhận checkpoint hash, config, epoch, threshold và đúng 203/395 predictions validation/test; metrics diagnosis được tính lại từ từng prediction trước khi tổng hợp.

Balanced Accuracy trung bình M3 (70.30%) gần M1 (70.74%), nhưng SD giữa seeds lớn hơn và ROC-AUC trung bình thấp hơn. Đây là so sánh mô tả; chưa thực hiện paired bootstrap để kết luận về chênh lệch.

Tối thiểu ba seeds được định trước: **42, 123, 2026**. Giữ nguyên split, preprocessing, augmentation, optimizer, epochs và checkpoint/threshold rules. Báo cáo tất cả seeds bằng mean ± sample SD; không chọn seed có test tốt nhất. Nếu đổi cấu hình huấn luyện, dùng một bộ runs mới cho toàn bộ seeds.

M1 hiện đã có đủ ba seeds, không cần chạy lại chỉ vì gộp lệnh. Với bộ thí nghiệm mới, mỗi lệnh tự train rồi test:

```bash
for m1_seed in 42 123 2026; do
  python3 experiments/run_m1.py --seed "$m1_seed" || break
done
```

M3 đã chốt và oracle MLP hiện đã có đủ ba seeds theo cùng danh sách. Tổng hợp M3 chỉ dùng bộ weighted đã chốt.

Sau khi có đủ ba exports M1:

```bash
python3 experiments/summarize_seeds.py \
  --results results/m1_efficientnet_b0_seed42_results.json \
            results/m1_efficientnet_b0_seed123_results.json \
            results/m1_efficientnet_b0_seed2026_results.json \
  --output_path results/m1_three_seeds_summary.json
```

Summarizer yêu cầu ít nhất ba seeds khác nhau, cùng cấu hình huấn luyện, cùng manifest/schema và đúng ca/nhãn test. Metrics diagnosis được tính lại từ predictions trước khi tổng hợp. SD dùng `ddof=1` giữa seeds; đây chưa phải CI theo mẫu hoặc kiểm định so sánh cặp. Ghi lại môi trường của từng run; ưu tiên cùng môi trường cho bộ thí nghiệm chính.

Với M3 sau khi có đủ ba exports test:

```bash
python3 experiments/summarize_seeds.py \
  --results results/m3_soft_joint_cbm_seed42_results.json \
            results/m3_soft_joint_cbm_seed123_results.json \
            results/m3_soft_joint_cbm_seed2026_results.json \
  --output_path results/m3_three_seeds_summary.json
```

### Kiểm tra code

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 tests/test_data_pipeline.py
```

Tests gồm 38 kiểm tra protocol, trong đó có regression mapping/LF–CRLF, lệnh train → test lưu một JSON cho cả bốn runners, giữ validation khi cập nhật test, từ chối gộp hai runs khác nhau, kiểm tra file đầu ra trước train, luồng weighted M3 → export → 128 intervention subsets bằng mạng nhỏ thay cho backbone, cùng oracle MLP và tổng hợp seeds. Các smoke tests dùng thư mục tạm; không tạo kết quả thực nghiệm chính. Chạy trong môi trường đã cài dependencies của dự án.

---

## 10. Tài liệu Tham khảo Chính (References)

1. Koh, P. W., et al. *"Concept Bottleneck Models."* International Conference on Machine Learning (ICML), 2020.
2. Kawahara, J., et al. *"Seven-Point Checklist and Skin Lesion Classification Using Multitask Multimodal Neural Nets."* IEEE Journal of Biomedical and Health Informatics (JBHI), 2019.
3. Nápoles, G., Grau, I. & Salgueiro, Y. *"Concept inconsistency in dermoscopic concept bottleneck models: a rough-set analysis of the Derm7pt dataset."* Scientific Reports (2026).
4. Patricio, C., et al. *"Coherent Concept-based Explanations for Skin Lesion Analysis."* (2023–2025).
