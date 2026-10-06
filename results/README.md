# Kết quả thí nghiệm

Các thư mục được phân theo **tiêu chí chọn checkpoint**, đọc từ metadata của run. Mỗi JSON vẫn có thể chứa cả BAcc và Macro F1 trong bảng metrics.

## Kết quả chính dùng cho khóa luận

**Bảng test chính là bảy model/cohort dưới đây**, theo các cấu hình đã chốt trong [Cấu hình đã chốt và thứ tự cải tiến](f1_macro/all_models_comparison.md#baseline-decisions). M3/M4 dùng cohort MLP128 20 epochs; các phiên bản Linear 10 epochs được lưu riêng để đối chiếu. Bảng này xác định cohort đã chốt, không chọn seed hoặc cấu hình theo điểm test cao nhất.

| Model | Cấu hình chính | Test Macro F1 | Nguồn số liệu |
|---|---|---:|---|
| M0 — Majority | Không huấn luyện | 0.4267 | [JSON nguồn](m0_majority_results.json) |
| M1 — Black-box | EfficientNet-B0, 10 epochs | 0.7078 ± 0.0118 | [JSON nguồn](f1_macro/m1/efficientnet_b0_e10/test_summary.json) |
| M2 — Oracle LR | Logistic Regression trên GT concepts | 0.8077 | [JSON nguồn](f1_macro/m2_lr/logistic_regression/seed42.json) |
| M2 — Oracle MLP | MLP trên GT concepts, 100 epochs | 0.8264 ± 0.0107 | [JSON nguồn](f1_macro/m2_mlp/mlp32_e100/test_summary.json) |
| M3 — Soft CBM | MLP128, 20 epochs | 0.7207 ± 0.0110 | [JSON nguồn](f1_macro/m3/mlp128_e20_headlr0.001/test_summary.json) |
| M4-ST — Hard CBM | MLP128, 20 epochs | 0.7131 ± 0.0071 | [JSON nguồn](f1_macro/m4_st/mlp128_e20_headlr0.001/test_summary.json) |
| M4-SG — Hard CBM | MLP128, 20 epochs | 0.7091 ± 0.0437 | [JSON nguồn](f1_macro/m4_sg/mlp128_e20_headlr0.001/test_summary.json) |

Test được đánh giá trên 395 ca. Các model có ba seeds báo cáo mean ± sample SD; M0 xác định và M2 LR chỉ có một run. M2 dùng nhãn concept thật nên là đối chứng oracle. Các checkpoint học được thuộc bộ F1; ngưỡng được chọn riêng trên validation bằng BAcc và giữ cố định khi test.

- **Metrics đầy đủ và nhận xét:** [all_models_comparison.md](f1_macro/all_models_comparison.md), đọc các dòng cohort chính ở bảng trên; báo cáo này còn trình bày phiên bản cũ để đối chiếu.
- **Bảng intervention riêng:** [interventions_comparison.md](f1_macro/interventions_comparison.md). Kết quả có GT concept hỗ trợ được báo cáo riêng với test tự động.
- **Kết quả lưu trữ/ablation:** `bacc/` và các cohort Linear 10 epochs. `f1_macro/all_models_comparison.md#legacy-f1` là bảng lịch sử của 16 runs đầu tiên.

## Năm báo cáo Markdown

| Cần đọc | File |
|---|---|
| Bảng chính và nguồn JSON | README này |
| Diagnosis, concept, train/validation, cấu hình đã chốt và cải tiến | [Báo cáo toàn model](f1_macro/all_models_comparison.md) |
| Intervention, từng seed/group và kết quả cũ | [Báo cáo intervention](f1_macro/interventions_comparison.md) |
| Study Linear/MLP128 của M4 | [Study head](bacc/ablation/m4_head_comparison/report.md) |
| Study LR diagnosis head M4-SG | [Study LR](bacc/ablation/m4_sg_head_lr_study/report.md) |

Các thông tin validation trước test và báo cáo riêng từng model được gộp vào các mục chi tiết có thể mở trong hai báo cáo chung. Số liệu JSON, CSV, PNG/SVG và weights giữ nguyên; đường dẫn JSON được cập nhật theo cấu trúc dưới đây. Runner intervention vẫn có thể tự sinh `report.md` khi chạy một thí nghiệm mới.

Đề cương và các báo cáo Word được cập nhật ngày **06/10/2026**: M0–M4 đã chốt làm bộ đối chứng; M5 chưa có code/kết quả model. Mục 10.1 của [đề cương](../docs/de-cuong-tong-hop.docx) thống nhất checkpoint theo validation diagnosis Macro F1@0.5, threshold theo validation BAcc. Mục 9.5 giữ M5 đầy đủ và hai ablations bắt buộc bỏ `E_global`/`E_class`; hai study M4 là thí nghiệm bổ sung. Giữ nguyên bộ BAcc lịch sử và các kết quả đã chốt.

## Cấu trúc và dữ liệu nguồn

| Thư mục | Nội dung hiện có | JSON |
|---|---|---:|
| [bacc/](#bacc) | Bộ BAcc lưu trữ/đối chiếu: baseline, summaries, intervention và ablation | 59 |
| [f1_macro/](#f1-macro) | Bộ kết quả chính: 25 runs có test, 8 summaries test, 3 summaries validation, 3 quyết định chốt và 12 JSON intervention M3/ST/SG MLP128 | 51 |

[M0 majority](m0_majority_results.json) dùng chung cho cả hai bộ vì không có checkpoint để chọn theo BAcc/F1.

[Bảng chính M0–M4](f1_macro/all_models_comparison.md) · [GT-assisted intervention ba cohorts](f1_macro/interventions_comparison.md). Bảng checkpoint BAcc cũ được giữ riêng; cột BAcc vẫn có trong bảng chính như một metric bổ sung.

M1, M2 LR/MLP, M3 và M4-ST/SG mặc định ghi JSON mới theo **tiêu chí → model → cấu hình → seed**. Truyền `--checkpoint_metric balanced_accuracy` để dùng `bacc/`. `--results_path` và `--output_dir` tùy chỉnh vẫn được ưu tiên.

```text
results/
├── m0_majority_results.json
├── f1_macro/
│   ├── m1/efficientnet_b0_e10/
│   ├── m2_lr/logistic_regression/
│   ├── m2_mlp/mlp32_e100/
│   ├── m3/
│   │   ├── linear_e10/
│   │   └── mlp128_e20_headlr0.001/
│   │       ├── seed42.json
│   │       ├── seed123.json
│   │       ├── seed2026.json
│   │       ├── test_summary.json
│   │       ├── validation_summary.json
│   │       ├── frozen_selection.json
│   │       ├── learning_curves.png / learning_curves.svg
│   │       └── intervention/
│   │           ├── seed42.json / seed123.json / seed2026.json
│   │           ├── summary.json
│   │           └── CSV, PNG và SVG
│   ├── m4_st/                 # cùng quy tắc với m3
│   └── m4_sg/                 # cùng quy tắc với m3
└── bacc/                      # cùng quy tắc; studies nằm trong ablation/
```

`mlp128` là diagnosis head MLP có hidden layer 128 units; `e20` là ngân sách train 20 epochs, không phải best epoch. `headlr0.001` là LR ban đầu của diagnosis head bằng 0.001; backbone/concept predictor dùng LR 0.0001 trong cohort này. Cấu hình khác augmentation, lambda concept loss, LR backbone hoặc weight decay có hậu tố riêng. Các model oracle có cấu hình riêng như `logistic_regression` và `mlp32_e100`.

| File | Dùng để đọc |
|---|---|
| `seed42.json`, `seed123.json`, `seed2026.json` | Kết quả train/validation/test của từng run |
| `test_summary.json` | Metrics test tổng hợp các seeds cùng cohort |
| `validation_summary.json` | Metrics validation của cohort đã chốt |
| `frozen_selection.json` | Quyết định cấu hình/checkpoint trước test |
| `intervention/seed*.json`, `intervention/summary.json` | Chi tiết và tổng hợp GT-assisted concept correction |

Intervention ba seeds nằm trong `<metric>/<model>/<cấu hình>/intervention/`. Các study BAcc cũ được lưu trong `bacc/ablation/<study>/<model>/<cấu hình>/`; plan, frozen selection và báo cáo chung của study ở tầng study. Khi gọi `summarize_seeds.py`, chọn inputs cùng protocol và đặt `--output_path` thành `test_summary.json` trong thư mục cấu hình tương ứng. M0 dùng file chung ở gốc; M2 LR chỉ có một run nên không có summary ba seeds.

Weights được chia tương ứng trong [checkpoints/bacc](../checkpoints/bacc/manifest.json) và [checkpoints/f1_macro](../checkpoints/f1_macro/README.md). Việc gom thư mục chỉ cập nhật đường dẫn tham chiếu; giữ nguyên metrics, predictions, cấu hình, thresholds và checkpoint hashes.


<a id="f1-macro"></a>
### Bộ F1 và cấu hình các nguồn kết quả

<details>
<summary>Mở bảng và ghi nhận chi tiết</summary>

Có **25 run JSONs**, tất cả đã hoàn tất train/validation/test: 16 baseline đầu tiên và ba seeds mỗi cohort M3/M4-ST/M4-SG MLP128 20 epochs. Có thêm tám summaries test, ba summaries validation và ba quyết định chốt trước test: **39 JSON trong các thư mục model/cấu hình**. Ba thư mục intervention M3/ST/SG MLP128 có tổng cộng chín run JSONs và ba summaries, đưa tổng số JSON kết quả lên **51**.

- [Chốt baseline, phần cần cải thiện và thứ tự công việc](f1_macro/all_models_comparison.md#baseline-decisions) (05/10/2026).
- [Rà soát tất cả mô hình, khoảng bất định và góp ý cải tiến](f1_macro/all_models_comparison.md) (cập nhật 06/10/2026; gồm test/intervention M0–M4 đã chốt, studies BAcc và ưu tiên M5).
- [Bảng baseline đầu tiên và nhận xét](f1_macro/all_models_comparison.md#legacy-f1).
- [M3 MLP128: train, validation và test ba seeds](f1_macro/all_models_comparison.md#m3-report).
- [M3 MLP128: intervention ba seeds và diễn giải](f1_macro/interventions_comparison.md#m3-analysis).
- [So sánh GT-assisted intervention M3/ST/SG MLP128](f1_macro/interventions_comparison.md).
- [M4-ST MLP128: intervention ba seeds](f1_macro/interventions_comparison.md#m4-st-details).
- [M4-SG MLP128: intervention ba seeds](f1_macro/interventions_comparison.md#m4-sg-details).
- [M3 MLP128: validation ba seeds và quyết định trước test](f1_macro/all_models_comparison.md#m3-validation).
- [M4-SG MLP128: train, validation và test ba seeds](f1_macro/all_models_comparison.md#m4-sg-report).
- [M4-SG MLP128: validation ba seeds và cấu hình đã chốt](f1_macro/all_models_comparison.md#m4-sg-validation).
- [M4-ST MLP128: pilot validation ba seeds và frozen checkpoints](f1_macro/all_models_comparison.md#m4-st-validation).
- [M4-ST MLP128: train, validation và frozen test ba seeds](f1_macro/all_models_comparison.md#m4-st-report).

Các JSON nguồn dùng tên `seed<seed>.json`; tiêu chí F1 đã thể hiện trong thư mục `f1_macro/`, nên không lặp hậu tố trong tên JSON. Không gộp cohort MLP128 20 epochs với baseline Linear 10 epochs; summaries phải dùng inputs cùng model, cấu hình và tiêu chí checkpoint. SD giữa seeds dùng `ddof=1`.

Runs chọn checkpoint theo `validation_f1_macro_at_0.5`. Ngưỡng quyết định vẫn được chọn riêng trên validation bằng BAcc; đây là quy tắc độc lập với việc chọn checkpoint. Cohort M4-SG mới đã chốt MLP128, 20 epochs, LR diagnosis head 1e-3 / backbone 1e-4 trước khi đánh giá test cả ba checkpoints. Test đã hoàn tất với mean Macro F1 **0.7091 ± 0.0437**; weights, best epochs, thresholds và validation giữ nguyên so với quyết định trước test. Chưa có kết quả M5 ECBM.

Cohort M3 MLP128 cùng ngân sách/head/LRs đã hoàn tất ba seeds: mean validation Macro F1@0.5 **0.7727 ± 0.0204**, mean test Macro F1 **0.7207 ± 0.0110**, BAcc **75.94% ± 2.07 điểm %**, AUC **0.8490 ± 0.0173**, concept F1 **0.4839 ± 0.0122**. Checkpoints epochs **19/7/18**, thresholds **0.072132/0.259394/0.090870**, weights và validation giữ nguyên so với quyết định trước test. Test chạy CPU; môi trường test và các đối chiếu được ghi trong [test summary](f1_macro/m3/mlp128_e20_headlr0.001/test_summary.json).

Intervention M3 mới đã hoàn tất **128 subsets × 395 ca × ba seeds**. Sửa đủ bảy groups bằng GT one-hot làm mean F1 **0.7207 → 0.6677**, BAcc **75.94% → 74.53%**; seed 123 giảm mạnh, seed 2026 tăng. Giữ kết quả này riêng với test tự động; weights và thresholds không đổi. [Báo cáo tự sinh](f1_macro/interventions_comparison.md#m3-details) · [Phân tích và đối chiếu](f1_macro/interventions_comparison.md#m3-analysis).

Những lệnh train mới dùng default F1 tự ghi JSON vào `f1_macro/<model>/<cấu hình>/`. Weights nằm trong [checkpoints/f1_macro](../checkpoints/f1_macro/README.md). Các hashes export validation trong quyết định chốt là snapshot trước test; bổ sung test hoặc đổi đường dẫn tham chiếu sẽ thay đổi bytes JSON nguồn nhưng phải giữ nguyên weights, best epoch, threshold và validation. Các hash snapshot trước test giữ nguyên để truy nguyên thời điểm chốt; chúng không phải checksum của JSON sau khi chuyển thư mục.

M4-ST MLP128 cùng head/LRs/ngân sách mới đã train đủ 20 epochs cho seeds 42/123/2026 trên MPS: mean validation F1@0.5 **0.7775 ± 0.0190**, F1 tại ngưỡng BAcc **0.7865 ± 0.0309**, AUC **0.7965 ± 0.0132**, concept F1 **0.5033 ± 0.0078**. Frozen test đã hoàn tất: Macro F1 **0.7131 ± 0.0071**, BAcc **72.66% ± 2.09 điểm %**, AUC **0.8302 ± 0.0092**, concept F1 **0.5069 ± 0.0015**. Best epochs **18/15/19**, thresholds **0.323596/0.427888/0.433949** và weights khớp [frozen selection](f1_macro/m4_st/mlp128_e20_headlr0.001/frozen_selection.json); toàn bộ JSON validation trước test tái tạo đúng SHA-256. Runtime train ST mới ghi PyTorch 2.12, M3/SG trước ghi 2.14; runner chưa lưu environment test riêng. Xem provenance và learning curves trong báo cáo validation, kiểm tra frozen test trong báo cáo mới.

Intervention ST và SG mới đã hoàn tất **128 subsets × 395 ca × ba seeds cho mỗi cohort** trên frozen CPU heads. Full GT correction ST tăng mean F1 **0.7131 → 0.7862 ± 0.0279**, BAcc **72.66% → 77.44%**; SG tăng F1 **0.7091 → 0.8212 ± 0.0228**, BAcc **72.53% → 83.43%**. Cả ba seeds đều tăng full F1 ở cả hai cohort; vẫn có ca/subset correction gây hại. GT-assisted metrics được giữ riêng với automatic test. Lượt kiểm tra 768 subsets mới đã xác minh head replay, official GT, no-op/remaining-group invariants, case transitions và aggregation; **147 artifacts có trước giữ nguyên SHA-256**. Baseline diagnosis/concept/intervention đã đủ để chuyển sang M5.

</details>


<a id="bacc"></a>
### Bộ BAcc lưu trữ và các nguồn dùng chung

<details>
<summary>Mở bảng và ghi nhận chi tiết</summary>

**Bộ lưu trữ/đối chiếu thử nghiệm bổ sung.** Bảng kết quả chính hiện tại dùng [checkpoint chọn theo Macro F1](f1_macro/all_models_comparison.md). Giữ nguyên bảng, JSON và weights BAcc để truy nguyên các studies LR/head; không gộp hai tiêu chí checkpoint vào cùng mean. Khi viết khóa luận, bảng BAcc có thể đặt ở phần ablation/phụ lục. BAcc vẫn là một metric phụ trong bảng kết quả chính F1.

59 JSON của các runs chọn checkpoint theo `validation_balanced_accuracy_at_0.5`, cùng metadata và phân tích đi kèm.

| Nhóm | Nội dung | JSON |
|---|---|---:|
| Các model/cấu hình baseline | 16 JSON baseline và 5 summaries | 21 |
| [m3/linear_e10/intervention/](f1_macro/interventions_comparison.md#bacc-m3) | M3: ba seeds và summary | 4 |
| [m4_st/linear_e10/intervention/](f1_macro/interventions_comparison.md#bacc-m4) | M4-ST: ba seeds và summary | 4 |
| [ablation/m4_head_comparison/](bacc/ablation/m4_head_comparison/report.md) | Thử Linear/MLP128, plan, lựa chọn và phân tích | 15 |
| [ablation/m4_sg_head_lr_study/](bacc/ablation/m4_sg_head_lr_study/report.md) | Thử LR head, plan, lựa chọn và phân tích; baseline 10 epochs dùng summary trong `m4_sg/linear_e10/` | 14 |
| [ablation/m4_evaluation/](f1_macro/all_models_comparison.md#legacy-bacc) | Đánh giá M4 | 1 |

Baseline gồm M1, M2 LR, M2 MLP, M3, M4-ST và M4-SG. Ba seeds là 42/123/2026; M2 LR có một run seed 42.

Study head dùng chung các summaries [M4-ST baseline](bacc/m4_st/linear_e10/test_summary.json), [M4-SG baseline](bacc/m4_sg/linear_e10/test_summary.json) và [SG Linear được chọn](bacc/ablation/m4_sg_head_lr_study/m4_sg/linear_e20_headlr0.001/test_summary.json); mỗi summary chỉ giữ một bản.

Weights nằm trong `../checkpoints/bacc/`; các trường `checkpoint_path`, `predictions_path` và `source_paths` trỏ đến vị trí hiện tại. CSV, report và hình của các nhóm con được giữ cùng nhóm.

Lưu ý có từ trước khi gom: JSON intervention M4 seed 2026 lưu hash của checkpoint cũ khác weights hiện tại. Kết quả và hash đó được giữ nguyên để bảo toàn nguồn gốc.

</details>
