# DermaXAI: Explainable AI for Skin Lesion Classification

## Trạng thái triển khai

Repo hiện có code và kết quả thực nghiệm **M0–M4**, gồm **M4-ST** và **M4-SG** qua ba seeds. **M5 categorical ECBM** vẫn là kế hoạch nghiên cứu.

| Mô hình | Trạng thái | Runner |
|---|---|---|
| M0 Majority baseline | Đã hoàn thành | `experiments/run_m0.py` |
| M1 EfficientNet-B0 | Đã hoàn thành, 3 seeds | `experiments/run_m1.py` |
| M2 Oracle LR / MLP | Đã hoàn thành; MLP có 3 seeds | `experiments/run_m2_lr.py`, `experiments/run_m2_mlp.py` |
| M3 Soft Joint CBM | Đã hoàn thành, 3 seeds | `experiments/run_m3.py` |
| M4-ST Hard Joint CBM | MLP128 đã có frozen test và intervention 3 seeds | `experiments/run_m4_st.py` |
| M4-SG Hard CBM | MLP128 đã có frozen test và intervention 3 seeds; giữ kết quả study LR cũ riêng | `experiments/run_m4_sg.py` |
| M5 Categorical ECBM | Dự kiến phân tích các thành phần năng lượng và nhóm concept bất nhất | Chưa triển khai |

`experiments/` có 13 file Python: các runner, intervention và ba module dùng chung (`m4_common.py`, `m4_interventions_common.py`, `summarize_seeds.py`). Mã điều phối hai study BAcc về LR/head đã được bỏ sau khi thí nghiệm hoàn tất; các JSON, báo cáo, biểu đồ và checkpoint cũ vẫn được giữ trong `results/bacc/ablation/` và `checkpoints/bacc/` để đối chiếu.

ECBM đã được dùng làm đối chứng trên Derm7pt trong [Wang et al., MIDL 2026](https://proceedings.mlr.press/v315/wang26a.html). Kế hoạch của dự án là đánh giá sâu hơn categorical ECBM trên official split; phần này chưa có kết quả để kết luận về hiệu quả hoặc tính mới.

## Báo cáo và đề cương

Các tài liệu Word cập nhật ngày **06/10/2026** đã thống nhất bộ đối chứng M0–M4, kết quả test/intervention và kế hoạch M5 cùng hai ablations bỏ `E_global`/`E_class`:

- [Đề cương tổng hợp](docs/de-cuong-tong-hop.docx).
- [Báo cáo tóm tắt tiến độ và kết quả](docs/bao-cao-tien-do.docx).
- [Báo cáo tóm tắt đề tài](docs/bao-cao-tom-tat-de-tai.docx).

Số liệu chi tiết nằm trong [danh mục kết quả](results/README.md). [Danh mục checkpoint và SHA-256](checkpoints/README.md) được lưu để đối chiếu; weights giữ ở máy cục bộ.

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

Xem [bảng kết quả chính M0–M4, chọn checkpoint theo Macro F1](results/f1_macro/all_models_comparison.md), [intervention ba cohorts MLP128](results/f1_macro/interventions_comparison.md) và [bản chốt baseline](results/f1_macro/all_models_comparison.md#baseline-decisions). Bảng checkpoint BAcc cũ ở mục 9 được giữ để lưu trữ/đối chiếu ablation; không gộp với bảng chính. Để chạy lại và giữ nguyên các kết quả đã công bố, ghi JSON mới vào `results/local/`:

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
| **3. Nút thắt** | Bottleneck $c$ | 7 nhóm concept (7-Point Checklist) | M3: xác suất soft 28 chiều; M4: one-hot 28 chiều |
| **4. Suy luận** | Mạng Chẩn đoán $g$ | Bộ phân loại Linear / MLP | Kết quả: 0 (Non-Melanoma) hoặc 1 (Melanoma); Non-Melanoma không đồng nghĩa lành tính |
| **5. Hiệu chỉnh** | Ground-truth intervention ($c^*$) | Thay group bằng annotation one-hot | Chạy lại diagnosis; kết quả có thể tốt lên hoặc xấu đi |

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

Mặc định (hoặc `--mode train`), M1, M2 LR, M2 MLP và M3 chạy toàn bộ quy trình bằng một lệnh. Bước huấn luyện chỉ dùng train/validation; checkpoint được chọn theo **diagnosis validation Macro F1 tại ngưỡng 0.5**. Nếu F1 hòa, giữ epoch đầu tiên. Ngưỡng quyết định được chọn riêng trên validation của checkpoint đó bằng cách tối đa Balanced Accuracy, hòa ưu tiên gần 0.5. Sau đó chương trình nạp lại checkpoint và ngưỡng đã đóng băng để đánh giá test; test không tham gia lựa chọn mô hình hoặc ngưỡng. M4-ST/SG dùng cùng tiêu chí mặc định này.

Checkpoint lưu theo tiêu chí trong `checkpoints/f1_macro/` hoặc `checkpoints/bacc/`; mỗi model/cấu hình/seed có **một JSON** tại `results/<metric>/<model>/<cấu hình>/seed<seed>.json`, chứa `validation_metrics`, `validation_predictions`, `test_metrics`, `test_predictions`, cấu hình và thông tin huấn luyện. Ví dụ `results/f1_macro/m1/efficientnet_b0_e10/seed42.json` và `results/f1_macro/m2_lr/logistic_regression/seed42.json`. Checkpoint F1 giữ hậu tố **`_ckptf1macro`**; JSON không lặp tiêu chí trong tên file. Metadata ghi `checkpoint_selection=validation_f1_macro_at_0.5`; history lưu riêng cả `validation_f1_macro_at_0.5` và `validation_balanced_accuracy_at_0.5`. `--results_path` tùy chỉnh file JSON này; đường dẫn phải khác checkpoint. Chương trình kiểm tra file đã tồn tại trước khi train; dùng `--overwrite` nếu muốn chạy lại. File được lưu sau bước validation, rồi cập nhật thêm test khi hoàn tất. `--mode test --overwrite` cập nhật phần test và giữ phần validation của cùng run đã lưu. `--no_save` (M1, M2 LR, M3) bỏ lưu JSON nhưng vẫn lưu checkpoint.

`--checkpoint_metric f1_macro` là mặc định mới cho M1, M2 LR/MLP, M3 và M4-ST/SG. Dùng `--checkpoint_metric balanced_accuracy` để tái lập protocol cũ và tìm các filenames cũ. Test/intervention đọc tiêu chí từ checkpoint đã đóng băng; không lựa chọn lại epoch trên test. Summarizers từ chối gộp runs có tiêu chí checkpoint khác nhau. Các bảng kết quả và hai studies tuning dưới đây là **kết quả BAcc đã chạy trước khi đổi protocol**; giữ nguyên provenance, không đổi nhãn thành F1. Mã điều phối study đã được bỏ; kết quả và weights BAcc vẫn lưu để đối chiếu. Chạy các runners train với default mới để tạo kết quả F1; chỉ sửa metadata hoặc chạy lại `--mode test` không khôi phục được weights của epoch F1 nếu epoch đó chưa được lưu.

34 checkpoint BAcc đã được gom vào **`checkpoints/bacc/`** (khoảng 1,37 GiB): `baseline/` chứa 16 bản chính, `m4_head_comparison/` chứa 9 bản thử head, và `m4_sg_head_lr_study/` chứa 9 bản thử LR head. Xem [danh sách checkpoint](checkpoints/README.md) và [manifest SHA-256](checkpoints/bacc/manifest.json). 59 JSON BAcc hiện nằm trong `results/bacc/`, gồm baseline, summaries, intervention và `ablation/`; bộ F1 hiện có 25 run JSONs đã có test, 8 summaries test, 3 summaries validation, 3 quyết định chốt và 12 JSON intervention M3/M4-ST/M4-SG MLP128 (51 JSON kết quả) trong `results/f1_macro/` ([tổ chức kết quả](results/README.md#f1-macro)). Xem [cách tổ chức kết quả](results/README.md). Các trường `checkpoint_path`, `predictions_path` và `source_paths` đã được cập nhật; metrics/predictions giữ nguyên. Khi truyền `--checkpoint_metric balanced_accuracy`, các runners tự tìm checkpoint chính trong `checkpoints/bacc/baseline/`. Weights của hai study BAcc vẫn nằm trong thư mục tương ứng, cùng plan và kết quả đã hoàn tất. 25 checkpoint F1 lưu trong **`checkpoints/f1_macro/`** với hậu tố `_ckptf1macro`; xem [tổ chức checkpoint](checkpoints/README.md), [danh mục F1 và hashes](checkpoints/f1_macro/manifest.json). Thư mục checkpoint tiếp tục được bỏ qua bởi Git.

---

## 7. Chạy M2 - Oracle Concept Model

**Mục đích:** Đánh giá concept sufficiency bằng classifier nhận bảy concept ground-truth, không dùng ảnh. LR (`class_weight="balanced"`) là baseline tuyến tính; oracle MLP nhỏ được báo cáo riêng để khảo sát tương tác phi tuyến. Kết quả phụ thuộc classifier, regularization và khả năng tổng quát hóa, không phải cận trên tuyệt đối. LR chọn C theo validation Macro F1@0.5, hòa giữ C đầu tiên trong grid, rồi chọn threshold BAcc bằng validation của model đã chọn.

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

Cũng như M1, bước huấn luyện chỉ dùng train/validation; checkpoint chọn theo diagnosis validation Macro F1 tại ngưỡng 0.5, ngưỡng cuối cùng chọn theo BAcc trên validation của checkpoint đó, sau đó tự động đánh giá test. Kết quả lưu thêm `validation_concept_metrics`/`test_concept_metrics` (Macro-F1 từng concept, cả "all-defined" và "train-observed") để phân tích riêng độ chính xác của tầng bottleneck $f: x \to c$, tách biệt khỏi độ chính xác chẩn đoán cuối $g: c \to y$.

Có thể chỉnh trọng số $\lambda$ giữa concept loss và diagnosis loss bằng `--concept_loss_weight` (mặc định `1.0`).

M3 được chốt là **một Soft Joint CBM với concept-state weighted CE**. State weights chỉ tính từ train, $w_s=N/(K_{observed}N_s)$; state không có mẫu train nhận weight 0. Batch đánh giá chỉ gồm states weight 0 có concept loss 0 và vẫn được tính trong metrics all-defined. CLI chỉ có một formulation; config lưu `protocol=soft_joint_state_weighted_v1` để xác nhận checkpoint thuộc pipeline đã chốt.

Export mới lưu GT/prediction/probabilities của từng concept theo case ID, exact-match, per-state F1/support và confusion matrix. Checkpoint M2/M3 lưu concept schema gồm mapping, order, cardinalities, offsets và hash; mapping phải nằm cạnh manifest. Hash manifest chuẩn hóa LF/CRLF nhưng vẫn kiểm tra thay đổi nội dung. M2/M3 được train lại bằng pipeline mới để tạo đầy đủ metadata. M3 test/intervention yêu cầu checkpoint đúng formulation đã chốt.

### Cấu hình M3 Linear đã chạy

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

Các lệnh dùng defaults trong bảng và chạy từ thư mục repo, trong môi trường đã cài dependencies. Device tự chọn CUDA, MPS hoặc CPU; có thể chỉ định `--device`. Tên artifact mặc định gồm model, seed và tiêu chí checkpoint, ví dụ `m3_soft_joint_cbm_seed42_ckptf1macro_best.pth`. Chạy mới tạo artifact riêng cho mỗi seed.

### Pilot M3 MLP128, LR head riêng, 20 epochs

Runner hỗ trợ `--diagnosis_head linear|mlp128`, `--diagnosis_lr` và `--skip_test`. MLP nhận vector soft 28 chiều qua `Linear(28,128) → LayerNorm → ReLU → Dropout(0.3) → Linear(128,2)`, cùng kiến trúc head như M4. M3 vẫn joint: diagnosis loss truyền gradient qua softmax về concept heads/backbone. Không truyền `--diagnosis_lr` thì toàn model dùng `--lr`; truyền LR riêng tách hai nhóm AdamW và history lưu LR từng nhóm trước mỗi epoch.

Cấu hình pilot: backbone/concept LR `1e-4`, diagnosis LR `1e-3`, weight decay `1e-2`, lambda `1.0`, batch `16`, augmentation `legacy_letterbox`, tối đa `20` epochs. Cấu hình này tạo cohort riêng để khảo sát baseline; train/validation cả ba seeds trước khi chốt để test:

```bash
for m3_seed in 42 123 2026; do
  python3 experiments/run_m3.py \
    --seed "$m3_seed" \
    --diagnosis_head mlp128 \
    --epochs 20 \
    --lr 0.0001 \
    --diagnosis_lr 0.001 \
    --weight_decay 0.01 \
    --checkpoint_metric f1_macro \
    --num_workers 0 \
    --skip_test || break
done
```

`--skip_test` chỉ hợp lệ với mode train; `--no_save` chỉ tắt JSON, checkpoint vẫn lưu. Cấu hình mới có đường dẫn riêng, ví dụ `checkpoints/f1_macro/m3_soft_joint_cbm_mlp128_headlr0.001_epochs20_seed42_ckptf1macro_best.pth` và `results/f1_macro/m3/mlp128_e20_headlr0.001/seed42.json`. Thay epoch, LR backbone, weight decay hoặc lambda cũng tạo tên riêng. Các checkpoint Linear F1 mặc định được nạp từ `checkpoints/f1_macro/`; kiến trúc và tên file giữ nguyên.

Pilot và frozen test đã hoàn tất seeds 42/123/2026: mean validation F1@0.5 **0.7727 ± 0.0204**; mean test F1 **0.7207 ± 0.0110**, BAcc **75.94% ± 2.07 điểm %**, AUC **0.8490 ± 0.0173**, concept F1 **0.4839 ± 0.0122**. Checkpoints epochs **19/7/18**, thresholds **0.072132/0.259394/0.090870**, weights và validation được giữ nguyên từ quyết định trước test. Xem [báo cáo train/validation/test](results/f1_macro/all_models_comparison.md#m3-report) và [báo cáo pilot validation](results/f1_macro/all_models_comparison.md#m3-validation).

Chọn cấu hình bằng mean validation diagnosis Macro F1@0.5 qua cả ba seeds, chốt checkpoint/thresholds trước khi đọc test. Ngưỡng vẫn tối ưu validation BAcc theo protocol hiện tại. Lệnh frozen test đã dùng cho cohort này (đã có kết quả; chỉ chạy lại khi cần tái tạo):

```bash
for m3_seed in 42 123 2026; do
  python3 experiments/run_m3.py --mode test --seed "$m3_seed" \
    --diagnosis_head mlp128 --diagnosis_lr 0.001 --epochs 20 --overwrite || break
done
```

Test/intervention tự dựng kiến trúc head và xác định JSON theo config checkpoint; `--checkpoint_path` tùy chỉnh không cần khai báo lại head. Test giữ validation/history và không đổi weights, best epoch hoặc threshold. Sau khi có test exports, intervention ba seeds của cohort mới dùng:

```bash
python3 experiments/run_m3_interventions.py \
  --diagnosis_head mlp128 --diagnosis_lr 0.001 --epochs 20
```

Thư mục mặc định là `results/f1_macro/m3/mlp128_e20_headlr0.001/intervention`, riêng với intervention Linear. Các tests nâng cấp dùng backbone nhỏ trong thư mục tạm để kiểm tra joint gradients, LR groups, pilot không đọc test, frozen MLP test và đủ 128 intervention subsets; đây không phải kết quả train Derm7pt.

Intervention của M3 MLP128 mới đã hoàn tất **128 subsets × 395 ca × ba seeds**. Sửa đủ bảy groups bằng annotation one-hot làm mean Macro F1 **0.7207 ± 0.0110 → 0.6677 ± 0.1296**, BAcc **75.94% ± 2.07 → 74.53% ± 9.76 điểm %**. Seed 123 giảm mạnh, seed 2026 tăng; lợi ích correction chưa ổn định. Baseline và tất cả 384 subsets đã được tái hiện từ frozen heads; weights, epochs, thresholds và test exports giữ nguyên. Xem [báo cáo intervention](results/f1_macro/interventions_comparison.md#m3-details) và [phân tích kết quả](results/f1_macro/interventions_comparison.md#m3-analysis). Kết quả có hỗ trợ GT concepts được báo cáo riêng với test tự động.

### Intervention M3

Chạy đủ ba seeds và xuất bảng/biểu đồ bằng một lệnh, dùng checkpoint và test predictions hiện có:

```bash
python3 experiments/run_m3_interventions.py --checkpoint_metric balanced_accuracy
```

Kết quả nằm trong [M3 BAcc cũ — intervention lưu trữ](results/f1_macro/interventions_comparison.md#bacc-m3): JSON chi tiết của mỗi seed, `summary.json`, `curve.csv`, `per_concept.csv`, `full_intervention_cases.csv`, biểu đồ PNG và SVG. Lệnh không train lại. Mặc định từ chối ghi đè; dùng `--output_dir` để tạo bộ kết quả riêng hoặc `--overwrite` để cập nhật bộ đã có. Chạy trong môi trường có dependencies của dự án.

Đã hoàn thành 128 subsets × 395 ca × ba seeds 42/123/2026. Baseline m=0 được tái hiện từ frozen head và khớp test export. Balanced Accuracy trung bình giảm từ **70.30% ± 4.58 điểm %** ở m=0 xuống **53.19% ± 2.83 điểm %** khi sửa đủ bảy groups; Macro-F1 giảm từ **0.6844 ± 0.0326** xuống **0.5262 ± 0.0243**. Đây là kết quả mô tả trên checkpoints hiện tại, chưa phải kết luận về nguyên nhân. Soft probabilities → GT one-hot thay đổi representation đầu vào của diagnosis head; ảnh hưởng của thay đổi này cần được khảo sát riêng. Kết quả chưa chứng minh concept leakage, inconsistency là nguyên nhân hoặc ECBM tốt hơn. M2 oracle được huấn luyện riêng nên không đồng nhất với M3 sau full intervention.

Nếu chỉ cần chạy một seed:

```bash
python3 experiments/run_m3.py --mode intervention --seed 42
python3 experiments/run_m3.py --mode intervention --seed 123
python3 experiments/run_m3.py --mode intervention --seed 2026
```

Train, test và intervention dùng chung `run_m3.py`. Intervention tự tìm checkpoint/export test theo seed và config; nếu dùng đường dẫn riêng thì truyền `--checkpoint_path` và `--predictions_path`. Evaluator đọc export test mới và head đã đóng băng, xác nhận hash checkpoint, config, schema, labels, case IDs và tái hiện baseline diagnosis. Duyệt đủ 128 subsets; thay cả group bằng GT one-hot, giữ groups chưa sửa và threshold cố định. Metrics được tính từng subset trước khi lấy trung bình theo m/7. Xuất diagnosis trước/sau, delta xác suất, ca được sửa đúng/xấu đi, độ đúng concepts chưa sửa (m=0…6), cùng intervention-curve AUC. SD giữa exhaustive subsets được ghi riêng với SD giữa seeds và CI theo mẫu. Đây là mô phỏng hiệu chỉnh annotation; tác động hiệu chỉnh chưa đo tầm quan trọng tổng quát hay quan hệ nhân quả.

---

## 8b. Chạy M4-ST — Hard Joint CBM

M4-ST giữ backbone, bảy concept heads và weighted loss như M3. Cấu hình baseline dùng diagnosis head Linear 28→2, ngân sách 10 epochs; preset mới hỗ trợ MLP128 và LR riêng cho diagnosis head. Diagnosis head nhận predicted one-hot ngay trong train. Backward dùng gradient xấp xỉ qua softmax ở temperature 1.0, không thêm nhiễu Gumbel; validation/test dùng deterministic argmax. M4 khởi tạo mới từ ImageNet, không dùng checkpoint M3 để warm-start. Runner ST là `run_m4_st.py`.

**Pilot MLP128, ba seeds:** cùng head/LR/ngân sách với cohort M3 và M4-SG mới: `Linear(28,128) → LayerNorm → ReLU → Dropout(0.3) → Linear(128,2)`, backbone/concept LR `1e-4`, head LR `1e-3`, weight decay `0.01`, batch 16, lambda 1.0, `legacy_letterbox`, 20 epochs, checkpoint theo validation Macro F1@0.5. Diagnosis gradient vẫn đi qua straight-through về concept/backbone. Chạy từ gốc repo:

```bash
for seed in 42 123 2026; do
  python3 -B -u experiments/run_m4_st.py \
    --seed "$seed" \
    --diagnosis_head mlp128 \
    --diagnosis_lr 0.001 \
    --epochs 20 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --batch_size 16 \
    --concept_loss_weight 1.0 \
    --augmentation_preset legacy_letterbox \
    --checkpoint_metric f1_macro \
    --device mps \
    --num_workers 0 \
    --skip_test || break
done
```

Vòng lặp gọi trực tiếp runner ST cho ba seeds, chỉ train/validation. Launcher pilot riêng đã được bỏ để tránh duy trì cấu hình trùng. Các artifacts dùng stem `m4_hard_joint_cbm_mlp128_headlr0.001_epochs20_seed{seed}_ckptf1macro`, riêng với Linear và các studies BAcc cũ. Runner từ chối ghi đè artifacts đã có; muốn chạy lại, dùng đường dẫn checkpoint/kết quả riêng.

Pilot mới đã hoàn tất ba seeds trên MPS: mean validation F1@0.5 **0.7775 ± 0.0190**, F1 tại ngưỡng BAcc **0.7865 ± 0.0309**, concept F1 **0.5033 ± 0.0078**. Best epochs **18/15/19**, thresholds **0.323596/0.427888/0.433949** đã đóng băng. Test và intervention cohort này đã hoàn tất, giữ nguyên weights/ngưỡng: mean test F1 **0.7131 ± 0.0071**; full GT-assisted intervention F1 **0.7862 ± 0.0279** được báo cáo riêng. [Báo cáo test](results/f1_macro/all_models_comparison.md#m4-st-report) · [So sánh intervention](results/f1_macro/interventions_comparison.md). Train diagnosis loss cuối lịch khoảng 0.213, validation khoảng 0.584; có dấu hiệu overfitting nên giữ best checkpoints và không tăng epoch. [Báo cáo validation, learning curves và provenance](results/f1_macro/all_models_comparison.md#m4-st-validation). Runtime ST mới dùng PyTorch 2.12, M3/SG trước ghi 2.14; phép so sánh cần giữ giới hạn này.

Sau khi chốt từ validation, test tự nạp kiến trúc từ checkpoint:

```bash
for m4_seed in 42 123 2026; do
  python3 experiments/run_m4_st.py --mode test --seed "$m4_seed" \
    --diagnosis_head mlp128 --diagnosis_lr 0.001 --epochs 20 --overwrite || break
done

python3 experiments/run_m4_st_interventions.py \
  --diagnosis_head mlp128 --diagnosis_lr 0.001 --epochs 20
```

Intervention preset mới ghi vào `results/f1_macro/m4_st/mlp128_e20_headlr0.001/intervention`. Head được kiểm tra tái hiện scores test trong sai số `atol=1e-6, rtol=1e-5`. Với ca có vector đầu vào diagnosis không đổi, evaluator giữ score test đã lưu để sai số làm tròn do batch size/device không gây đổi nhãn sát ngưỡng; policy và sai số replay được ghi trong JSON. Các ca thực sự đổi vector được chạy lại qua frozen head ở CPU, với ngưỡng đã chốt.

**Pilot seed 42:** chạy train/validation, chưa đọc test. Lưu trong thư mục local để phân biệt pilot với kết quả chính thức:

```bash
python3 experiments/run_m4_st.py --seed 42 --skip_test --num_workers 0 \
  --checkpoint_path checkpoints/local/m4_pilot_seed42_best.pth \
  --results_path results/local/m4_pilot_seed42_results.json
```

`--skip_test` chỉ dùng với mode train. `--no_save` chỉ tắt lưu JSON; lệnh vẫn chạy test nếu không có `--skip_test`, và checkpoint vẫn được tạo.

**Ba seeds chính thức sau khi chốt cấu hình:** mỗi lệnh mặc định chạy train → chọn checkpoint/ngưỡng trên validation → test, giống M1–M3.

```bash
for m4_seed in 42 123 2026; do
  python3 experiments/run_m4_st.py --seed "$m4_seed" --num_workers 0 || break
done

python3 experiments/summarize_seeds.py \
  --results results/f1_macro/m4_st/linear_e10/seed42.json \
            results/f1_macro/m4_st/linear_e10/seed123.json \
            results/f1_macro/m4_st/linear_e10/seed2026.json \
  --output_path results/f1_macro/m4_st/linear_e10/test_summary.json

python3 experiments/run_m4_st_interventions.py
```

Default artifacts dùng checkpoint `m4_hard_joint_cbm_seed{seed}_ckptf1macro_best.pth` và JSON `results/f1_macro/m4_st/linear_e10/seed{seed}.json`. Các cấu hình khác augmentation/lambda có tên riêng. Lệnh từ chối ghi đè file đã có; dùng đường dẫn mới hoặc `--overwrite` khi chủ động chạy lại. Nếu một seed thất bại, hoàn tất seed đó và các seeds còn thiếu trước khi tổng hợp.

M4 giữ mode test cho checkpoint có sẵn. Ví dụ hoàn tất test của pilot khi cấu hình pilot đã được chốt làm cấu hình chính thức:

```bash
python3 experiments/run_m4_st.py --mode test --num_workers 0 \
  --checkpoint_path checkpoints/local/m4_pilot_seed42_best.pth \
  --results_path results/local/m4_pilot_seed42_results.json --overwrite
```

Validation vẫn được giữ khi bổ sung test vào cùng JSON. Khi tái sử dụng checkpoint pilot hoặc dùng đường dẫn riêng, truyền đủ checkpoint/test export của từng seed cho batch intervention:

```bash
python3 experiments/run_m4_st_interventions.py \
  --checkpoints checkpoints/local/m4_pilot_seed42_best.pth \
                checkpoints/f1_macro/m4_hard_joint_cbm_seed123_ckptf1macro_best.pth \
                checkpoints/f1_macro/m4_hard_joint_cbm_seed2026_ckptf1macro_best.pth \
  --predictions results/local/m4_pilot_seed42_results.json \
                results/f1_macro/m4_st/linear_e10/seed123.json \
                results/f1_macro/m4_st/linear_e10/seed2026.json \
  --output_dir results/local/m4_intervention
```

Trong trường hợp này, thay đường dẫn seed 42 tương ứng trong lệnh tổng hợp diagnosis. Mọi run phải cùng cấu hình huấn luyện, schema và official cases; không gộp pilot có cấu hình khác vào bộ chính thức.

Export M4 lưu soft probabilities để kiểm tra concept và `hard_concept_vector` là đầu vào thực tế của diagnosis head. Summarizer kiểm tra hard vectors/argmax/annotation và tính lại concept metrics ngoài diagnosis metrics. Evaluator từ chối checkpoint M3 hoặc sai hard/ST protocol. Intervention bắt đầu từ hard predictions, duyệt 128 subsets với head/ngưỡng đóng băng, xác nhận baseline khớp paired test export. Xuất JSON, CSV, PNG/SVG và report M4 riêng tại `results/f1_macro/m4_st/linear_e10/intervention/` cho runs F1 mới; báo cáo BAcc cũ nằm tại `results/bacc/m4_st/linear_e10/intervention`.

Các smoke tests dùng mạng nhỏ trong thư mục tạm để kiểm tra pipeline, không phải kết quả nghiên cứu Derm7pt. Kết quả M4-ST ba seeds đã lưu ở `results/bacc/m4_st/linear_e10/test_summary.json`, đánh giá tại `results/f1_macro/all_models_comparison.md#legacy-bacc`.

## 8c. Chạy M4-SG — Hard CBM với Stop-Gradient

M3, M4-ST và M4-SG được định nghĩa chung trong `src/cbm.py`; hai biến thể M4 có runners riêng:

| Biến thể | Model | Train / test / intervention | Tổng hợp intervention |
|---|---|---|---|
| M4-ST | `src/cbm.py` (`HardJointCBM`) | `experiments/run_m4_st.py` | `experiments/run_m4_st_interventions.py` |
| M4-SG | `src/cbm.py` (`HardStopGradientCBM`) | `experiments/run_m4_sg.py` | `experiments/run_m4_sg_interventions.py` |

Mỗi runner cố định một biến thể; các lệnh mới không có flag `--gradient_mode`. Các lớp CBM cùng backbone/concept heads nằm trong `src/cbm.py`, còn quy trình train/test và báo cáo dùng chung helpers để giữ phép so sánh nhất quán. Import cũ từ `src/models.py` vẫn được hỗ trợ; dùng `run_m4_st.py` cho ST và `run_m4_sg.py` cho SG.

M4-SG chỉ thay luồng gradient so với M4-ST: one-hot bottleneck được detach trước diagnosis head. Loss chẩn đoán cập nhật diagnosis head; concept loss cập nhật concept heads và backbone. Cả hai loss vẫn được tối ưu trong cùng bước train. Giữ nguyên EfficientNet-B0, Linear 28→2, preprocessing, weighted loss, LR, dropout, 10 epochs và tiêu chí checkpoint/ngưỡng validation để đối chiếu riêng cơ chế gradient.

Đây là ablation của M4-ST, chưa phải bản tái lập đầy đủ Nápoles (paper dùng diagnosis MLP128 và pipeline huấn luyện khác). M5 vẫn dành cho ECBM.

Chạy seed 42 theo cấu hình cố định (train → validation → frozen test):

```bash
python3 experiments/run_m4_sg.py --seed 42 --num_workers 0
```

Ba seeds và tổng hợp diagnosis/intervention:

```bash
for m4_sg_seed in 42 123 2026; do
  python3 experiments/run_m4_sg.py --seed "$m4_sg_seed" --num_workers 0 || break
done

python3 experiments/summarize_seeds.py \
  --results results/f1_macro/m4_sg/linear_e10/seed42.json \
            results/f1_macro/m4_sg/linear_e10/seed123.json \
            results/f1_macro/m4_sg/linear_e10/seed2026.json \
  --output_path results/f1_macro/m4_sg/linear_e10/test_summary.json

python3 experiments/run_m4_sg_interventions.py
```

Nếu đã chạy riêng seed 42, chạy tiếp seeds 123 và 2026; lệnh mặc định từ chối ghi đè kết quả có sẵn. Với pilot chỉ train/validation, thêm `--skip_test` và dùng đường dẫn local riêng như M4-ST. Chốt mọi lựa chọn/tuning bằng validation trước khi đọc test.

SG có model `M4_HardStopGradientCBM`, protocol `hard_joint_sg_state_weighted_v1`, checkpoint `checkpoints/f1_macro/m4_hard_sg_cbm_seed{seed}_ckptf1macro_best.pth` và JSON `results/f1_macro/m4_sg/linear_e10/seed{seed}.json`. Batch intervention mới xuất riêng vào `results/f1_macro/m4_sg/linear_e10/intervention/`. Checkpoint/export ST và SG không được ghép hoặc gộp chung seeds.

Runner ST và SG chỉ chấp nhận checkpoint của đúng biến thể tương ứng. Chạy frozen test và intervention SG:

```bash
python3 experiments/run_m4_sg.py --mode test --seed 42 --overwrite
python3 experiments/run_m4_sg.py --mode intervention --seed 42
```

Tests SG kiểm tra diagnosis loss không cập nhật concept heads/backbone, concept loss vẫn cập nhật chúng, hard forward và khởi tạo khớp ST, cũng như train/test, checkpoint identity, tổng hợp seeds và 128 intervention subsets.

### Thử learning rate riêng cho diagnosis head M4-SG

`--lr` điều khiển backbone/concept heads; `--diagnosis_lr` tách LR của diagnosis head thành optimizer group riêng. Không truyền `--diagnosis_lr` thì giữ optimizer cũ. Cấu hình có LR head riêng lưu thêm `diagnosis_learning_rate`, optimizer groups và LR từng epoch; đường dẫn mặc định có hậu tố head LR/epochs để tách khỏi baseline. Checkpoint cũ vẫn được hỗ trợ.

Ví dụ một pilot, chỉ dùng train/validation:

```bash
python3 experiments/run_m4_sg.py --diagnosis_lr 0.001 --epochs 20 --seed 42 --skip_test
```

**Study BAcc đã hoàn tất:** grid head LR `1e-4`, `5e-4`, `1e-3`; mỗi mức dùng 20 epochs và seeds 42/123/2026, concept predictor LR `1e-4`, weight decay `1e-2`. LR được chọn theo mean validation diagnosis BAcc@0.5, hòa ưu tiên LR nhỏ hơn. Quyết định đã được ghi vào `frozen_selection.json` trước khi đánh giá test đối chứng 20 epochs `1e-4` và LR được chọn; ngưỡng test lấy từ validation riêng của từng checkpoint.

Plan, JSON từng run, summaries, report và biểu đồ vẫn lưu trong `results/bacc/ablation/m4_sg_head_lr_study/`; weights trong `checkpoints/bacc/m4_sg_head_lr_study/`. Script điều phối/phân tích study đã được bỏ. Bộ tuning này được giữ để đối chiếu và không gộp với M4-SG 10 epochs hoặc các cohort chọn checkpoint F1.

Đợt thử đã hoàn tất 9 runs. Validation chọn head LR **1e-3** (concept predictor LR **1e-4**, 20 epochs). Test BAcc đạt **71.59% ± 1.99 điểm %**, Macro-F1 **0.6868 ± 0.0259**, AUC **0.8016 ± 0.0141**. Đối chứng 20 epochs cùng môi trường, head LR 1e-4, đạt BAcc **61.78% ± 1.87 điểm %**. Concept loss của ba mức LR khớp hoàn toàn ở cùng seed/epoch; LR riêng thay cách học diagnosis head. Đây là đợt tuning thăm dò riêng, chưa thay bảng thí nghiệm chính M0–M3.

Chi tiết, learning curves, kiểm tra concept-loss trajectories và paired stratified bootstrap đã lưu trong [báo cáo study LR](results/bacc/ablation/m4_sg_head_lr_study/report.md).

### M4-ST/SG: thử head MLP128 và train ST với LR head riêng

Cả hai runners hỗ trợ `--diagnosis_head linear` (mặc định) hoặc `--diagnosis_head mlp128`. MLP chỉ nhận 28 chiều concept, theo cấu trúc `Linear(28,128) → LayerNorm → ReLU → Dropout(0.3) → Linear(128,2)`. ST vẫn dùng straight-through; SG vẫn chặn diagnosis gradient về concept predictor. Checkpoint lưu kiến trúc head, và test/intervention dựng lại đúng head từ cấu hình đó. Tên file MLP có hậu tố riêng.

Ví dụ pilot M4-ST MLP, chỉ train/validation:

```bash
python3 experiments/run_m4_st.py --diagnosis_head mlp128 --diagnosis_lr 0.001 --epochs 20 --seed 42 --skip_test
```

**Study BAcc so sánh head đã hoàn tất:** dùng 20 epochs, head LR `1e-3`, backbone/concept LR `1e-4`, weight decay `1e-2` và `legacy_letterbox`. Ba SG Linear runs đã được tái sử dụng từ study LR; ST Linear, ST MLP và SG MLP có chín runs train mới. Head được chọn riêng cho ST/SG bằng mean validation BAcc@0.5, hòa giữ Linear. Lựa chọn đã được ghi vào `frozen_selection.json` trước khi test Linear controls và heads được chọn. Mã điều phối/phân tích study đã được bỏ; checkpoint, JSON, summaries, báo cáo và biểu đồ cũ vẫn được giữ.

Kết quả lưu trong `results/bacc/ablation/m4_head_comparison/`, tách khỏi baseline 10 epochs. Đây là thử một thay đổi kiến trúc với protocol khóa luận; chưa phải tái lập toàn bộ Nápoles. Mốc đối chiếu cùng raw Derm7pt / EfficientNet-B0 là **HardCBM test Macro F1 0.73 ± 0.03**, không phải accuracy/BAcc; paper dùng 5 seeds, preprocessing/augmentation khác và chọn checkpoint bằng validation Macro F1. [Nápoles et al., Scientific Reports (2026)](https://www.nature.com/articles/s41598-026-56927-2).

Đợt so sánh đã hoàn tất chín runs mới và tái sử dụng ba SG Linear runs. Validation chọn MLP cho cả ST/SG, nhưng test không xác nhận tăng BAcc: ST Linear **74.15% ± 2.26**, ST MLP **72.59% ± 1.97**; SG Linear **71.59% ± 1.99**, SG MLP **71.50% ± 1.59**. Cả hai paired case-bootstrap CI 95% của chênh lệch MLP–Linear đều chứa 0. SG MLP tăng Macro F1 quan sát từ **0.6868** lên **0.7026**; vẫn thấp hơn mean **0.73** của paper. ST Linear đạt Macro F1 **0.7250**, nhưng ST có gradient khác HardCBM stop-gradient của paper.

Kết quả này hỗ trợ việc train lại ST với LR head riêng/20 epochs so với baseline cũ; chưa đủ để đổi head chính sang MLP nhằm cải thiện BAcc. Default vẫn là baseline Linear 10 epochs; để chạy ST Linear với cấu hình thử mới, dùng `--diagnosis_lr 0.001 --epochs 20`. Đây chưa phải LR tối ưu đã được tune riêng cho ST. Báo cáo, kiểm tra 12 checkpoints/thresholds và biểu đồ: [`results/bacc/ablation/m4_head_comparison/report.md`](results/bacc/ablation/m4_head_comparison/report.md).

---

## 9. Bảng lưu trữ M0–M3: checkpoint chọn theo BAcc

**Bảng cũ để đối chiếu, không phải bảng kết quả chính hiện tại.** Các runs dưới đây chọn checkpoint theo validation BAcc@0.5 và được lưu trong `results/bacc/`; M0 là mốc dùng chung. [Bảng chính chọn checkpoint Macro F1](results/f1_macro/all_models_comparison.md) đã có M3/M4-ST/M4-SG MLP128 20 epochs và báo cáo riêng [GT-assisted intervention](results/f1_macro/interventions_comparison.md). Khi viết khóa luận, dùng bảng F1 ở phần kết quả chính; giữ bảng BAcc ở phần thử nghiệm bổ sung/phụ lục. Cột metric BAcc vẫn có thể báo cáo trong bảng F1, vì metric và tiêu chí chọn checkpoint là hai việc khác nhau.

M1, M2 MLP và M3 dùng ba seeds 42, 123, 2026, báo cáo mean ± sample SD. M0 là baseline xác định; M2 LR dùng một seed với solver lbfgs. SD của Balanced Accuracy có đơn vị điểm phần trăm. Hai exports M3 unweighted cũ đã được xoá sau khi hoàn thành bộ M3 weighted.

| Model | Balanced Acc | Macro-F1 | ROC-AUC | AP | Runs |
|:---|:---:|:---:|:---:|:---:|:---:|
| **M0** Majority Baseline | 50.00% | 0.4267 | 0.5000 | 0.2557 | 1 |
| **M1** Black-box EfficientNet-B0 | 70.74% ± 1.88 | 0.7078 ± 0.0118 | 0.8302 ± 0.0062 | 0.6608 ± 0.0115 | 3 seeds |
| **M2 LR** Oracle trên GT concepts | 85.37% | 0.8077 | 0.9211 | 0.7848 | 1 seed |
| **M2 MLP** Oracle trên GT concepts | 86.23% ± 1.27 | 0.8220 ± 0.0154 | 0.9246 ± 0.0029 | 0.7981 ± 0.0043 | 3 seeds |
| **M3** Weighted Soft Joint CBM | 70.30% ± 4.58 | 0.6844 ± 0.0326 | 0.7616 ± 0.0446 | 0.5917 ± 0.0240 | 3 seeds |

**Nhận xét mô tả:** Hai oracle M2 nhận GT concepts và đạt Balanced Accuracy cao hơn các models dùng ảnh trong bộ runs này. M2 MLP cao hơn LR khoảng 0.86 điểm phần trăm; M1 cao hơn M3 weighted khoảng 0.44 điểm phần trăm. Chưa thực hiện paired bootstrap để kết luận về chênh lệch. Khoảng cách M2–M3 chưa tự xác lập nguyên nhân hoặc chứng minh inconsistency. Lợi ích ECBM được kết luận sau thực nghiệm, kể cả trường hợp không cải thiện.

### M1: ba seeds đã hoàn thành

M1 đã hoàn thành train/validation và test cho seeds 42, 123, 2026 với cấu hình 10 epochs, batch 16, LR 1e-4, weight decay 1e-2, legacy_letterbox. Checkpoint tốt nhất lần lượt ở epochs 7, 7, 8; threshold lấy từ validation của từng checkpoint. Các metrics test đã được tính lại từ prediction records và tổng hợp trong [test_summary.json của M1](results/bacc/m1/efficientnet_b0_e10/test_summary.json).

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

Concept accuracy trung bình là **59.61% ± 2.50 điểm %**, mean concept Macro-F1 all-defined **0.4570 ± 0.0185**, exact match cả bảy concepts **4.22% ± 0.81 điểm %**. Kết quả lưu trong [test_summary.json của M3 Linear](results/bacc/m3/linear_e10/test_summary.json). Đã xác nhận checkpoint hash, config, epoch, threshold và đúng 203/395 predictions validation/test; metrics diagnosis được tính lại từ từng prediction trước khi tổng hợp.

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
  --results results/bacc/m1/efficientnet_b0_e10/seed42.json \
            results/bacc/m1/efficientnet_b0_e10/seed123.json \
            results/bacc/m1/efficientnet_b0_e10/seed2026.json \
  --output_path results/bacc/m1/efficientnet_b0_e10/test_summary.json
```

Summarizer yêu cầu ít nhất ba seeds khác nhau, cùng cấu hình huấn luyện, cùng manifest/schema và đúng ca/nhãn test. Metrics diagnosis được tính lại từ predictions trước khi tổng hợp. SD dùng `ddof=1` giữa seeds; đây chưa phải CI theo mẫu hoặc kiểm định so sánh cặp. Ghi lại môi trường của từng run; ưu tiên cùng môi trường cho bộ thí nghiệm chính.

Với M3 sau khi có đủ ba exports test:

```bash
python3 experiments/summarize_seeds.py \
  --results results/bacc/m3/linear_e10/seed42.json \
            results/bacc/m3/linear_e10/seed123.json \
            results/bacc/m3/linear_e10/seed2026.json \
  --output_path results/bacc/m3/linear_e10/test_summary.json
```

### Kiểm tra code

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 tests/test_data_pipeline.py
```

Tests kiểm tra mapping/LF–CRLF, bảo vệ artifact, train → frozen test, giữ validation khi re-export, hard forward, gradient ST/SG, `--skip_test`/`--no_save`, M4 checkpoint identity, hard export và concept metric integrity, 128 intervention subsets, tổng hợp ba seeds và regression report M3. Các smoke tests dùng mạng nhỏ trong thư mục tạm; không tạo kết quả thực nghiệm chính. Chạy trong môi trường đã cài dependencies của dự án.

Tên file test chỉ rõ biến thể: M2 dùng `tests/test_m2_lr_protocol.py` và `tests/test_m2_mlp_protocol.py`; M4 dùng `tests/test_m4_st_protocol.py` và `tests/test_m4_sg_protocol.py`. Các kiểm tra chung giữa nhiều runners nằm trong `tests/test_protocol_safety.py`.

---

## 10. Tài liệu Tham khảo Chính (References)

1. Koh, P. W., et al. *"Concept Bottleneck Models."* International Conference on Machine Learning (ICML), 2020.
2. Kawahara, J., et al. *"Seven-Point Checklist and Skin Lesion Classification Using Multitask Multimodal Neural Nets."* IEEE Journal of Biomedical and Health Informatics (JBHI), 2019.
3. Nápoles, G., Grau, I. & Salgueiro, Y. *"Concept inconsistency in dermoscopic concept bottleneck models: a rough-set analysis of the Derm7pt dataset."* Scientific Reports (2026).
4. Patricio, C., et al. *"Coherent Concept-based Explanations for Skin Lesion Analysis."* (2023–2025).
