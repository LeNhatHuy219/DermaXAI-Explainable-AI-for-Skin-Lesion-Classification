"""Tiêu chí chọn model/checkpoint thống nhất, độc lập với việc chọn ngưỡng."""
# Các runners M1–M4 dùng chung quy tắc ở đây để chọn checkpoint nhất quán.
# Runner tính metrics trên validation mỗi epoch, lấy điểm qua selection_score(),
# rồi so sánh với điểm tốt nhất để quyết định lưu checkpoint.
import math
from pathlib import Path

# Mặc định mới: trung bình F1 của hai lớp melanoma và non-melanoma.
# Balanced Accuracy được giữ lại để có thể tái lập các thí nghiệm theo tiêu chí cũ.
DEFAULT_CHECKPOINT_METRIC = "f1_macro"
CHECKPOINT_METRICS = ("f1_macro", "balanced_accuracy")


def selection_name(metric):
    # Kiểm tra tên metric trước khi tạo chuỗi mô tả dùng trong metadata checkpoint.
    if metric not in CHECKPOINT_METRICS:
        raise ValueError("checkpoint_metric must be f1_macro or balanced_accuracy")
    # @0.5 là ngưỡng tính metric lúc chọn epoch trên validation.
    # Ngưỡng dùng cho test/intervention được chọn riêng trên validation sau đó.
    return f"validation_{metric}_at_0.5"


def selection_score(metrics, metric):
    # Chọn theo diagnosis Macro F1, không nhầm với F1 melanoma/concept F1.
    # Xác nhận metric hợp lệ, rồi lấy điểm từ dict metrics của epoch đang xét.
    selection_name(metric)
    score = float(metrics[metric])
    # Loại bỏ NaN, vô cực và điểm ngoài [0, 1] để tránh lưu checkpoint sai.
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Checkpoint selection score must be finite and in [0, 1]")
    # Runner dùng điểm này để so sánh; nếu hòa thì giữ epoch tốt nhất đầu tiên.
    return score


def metric_from_config(config):
    # Artifact cũ có BAcc (hoặc thiếu metadata) vẫn được đọc theo protocol cũ.
    # Tiêu chí được đọc từ config đã lưu, thay vì lấy mặc định F1 của lần chạy mới.
    name = config.get("checkpoint_selection", selection_name("balanced_accuracy"))
    for metric in CHECKPOINT_METRICS:
        if name == selection_name(metric):
            # Nếu config có cả hai trường thì chúng phải mô tả cùng một tiêu chí.
            if config.get("checkpoint_metric", metric) != metric:
                raise ValueError("Checkpoint metric and selection metadata differ")
            return metric
    # Metadata lạ phải được kiểm tra lại, thay vì tự đoán tiêu chí chọn checkpoint.
    raise ValueError("Unknown checkpoint selection metadata")


def selection_suffix(metric):
    # Tên riêng cho F1 để không trộn checkpoint/kết quả mới với các runs BAcc.
    selection_name(metric)
    # BAcc giữ tên weights cũ; F1 thêm hậu tố trước phần _best.pth/.joblib.
    return "_ckptf1macro" if metric == "f1_macro" else ""


def checkpoint_directory(project_root, metric):
    """Lưu weights theo tiêu chí chọn checkpoint đã dùng trên validation."""
    selection_name(metric)
    directory = Path(project_root) / "checkpoints"
    return directory / "bacc" / "baseline" if metric == "balanced_accuracy" else directory / "f1_macro"


def results_directory(project_root, metric):
    """Đường dẫn kết quả theo tiêu chí đã dùng để chọn checkpoint."""
    selection_name(metric)
    folder = "bacc" if metric == "balanced_accuracy" else "f1_macro"
    return Path(project_root) / "results" / folder


# Tên thư mục cho một cấu hình; seed nằm trong tên JSON để ba runs ở cùng cohort.
def configuration_name(head, epochs, diagnosis_lr=None, *, augmentation_preset="legacy_letterbox",
                       concept_loss_weight=1., lr=1e-4, default_lr=1e-4, weight_decay=.01):
    name = f"{head}_e{epochs}"
    if diagnosis_lr is not None:
        name += f"_headlr{diagnosis_lr:g}"
    if augmentation_preset != "legacy_letterbox":
        name += f"_{augmentation_preset}"
    if concept_loss_weight != 1.:
        name += f"_lambda{concept_loss_weight:g}"
    if lr != default_lr:
        name += f"_lr{lr:g}"
    if weight_decay != .01:
        name += f"_wd{weight_decay:g}"
    return name


# Đường dẫn thống nhất: results/<metric>/<model>/<cấu hình>/seed<seed>.json.
def results_run_path(project_root, metric, model, configuration, seed):
    return results_directory(project_root, metric) / model / configuration / f"seed{seed}.json"


def add_selection_argument(parser):
    # Thêm tùy chọn chung cho CLI, ví dụ: --checkpoint_metric f1_macro.
    # argparse giới hạn hai tên hợp lệ; bỏ qua tùy chọn này thì mặc định dùng F1.
    parser.add_argument("--checkpoint_metric", choices=CHECKPOINT_METRICS,
        default=DEFAULT_CHECKPOINT_METRIC,
        help="Diagnosis metric trên validation @0.5 để chọn checkpoint; hòa giữ epoch đầu")
