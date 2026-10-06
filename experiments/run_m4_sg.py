# Runner M4-SG - Hard CBM với stop-gradient ở concept bottleneck.
# Ảnh -> concept logits -> hard one-hot 28 chiều -> diagnosis Linear hoặc MLP128.
# Diagnosis gradient dừng ở hard concepts; concept/backbone vẫn học từ concept loss.
# Các hàm dưới đây gọi m4_common.py với gradient_mode="stop_gradient".
# CLI hỗ trợ train/test/intervention; --skip_test giữ run ở train/validation.
# Đầu ra JSON: results/<metric>/m4_sg/<cấu hình>/seed<seed>.json; checkpoint giữ tên cũ.
# BAcc/path riêng theo tham số; test/intervention phải dùng checkpoint đúng variant SG.

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import m4_common as common
from experiments.m4_common import (DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY, DEFAULT_SEED, DEFAULT_AUGMENTATION_PRESET,
    DEFAULT_CONCEPT_LOSS_WEIGHT, _protect_inputs, evaluate)

M4_MODEL = common.M4_SG_MODEL
M4_PROTOCOL = common.M4_SG_PROTOCOL


# Tạo checkpoint/JSON paths theo cấu hình, seed và tiêu chí chọn checkpoint.
def _default_paths(seed=DEFAULT_SEED, augmentation_preset=DEFAULT_AUGMENTATION_PRESET,
                   concept_loss_weight=DEFAULT_CONCEPT_LOSS_WEIGHT,
                   checkpoint_metric=common.DEFAULT_CHECKPOINT_METRIC):
    return common._default_paths(seed, augmentation_preset, concept_loss_weight, "stop_gradient",
                                 checkpoint_metric=checkpoint_metric)


# Kiểm tra config M4 và yêu cầu variant SG trước khi dùng checkpoint.
def validate_m4_sg_config(config):
    criteria = common.validate_m4_config(config)
    common._require_gradient_mode(config, "stop_gradient")
    return criteria


# Gọi train/validation chung nhưng cố định gradient_mode stop_gradient.
def run_m4_sg_experiment(*args, **kwargs):
    return common.run_m4_experiment(*args, **kwargs, gradient_mode="stop_gradient")


# Gọi frozen test chung, yêu cầu checkpoint SG; không chọn lại epoch/ngưỡng.
def evaluate_m4_sg_test(*args, **kwargs):
    return common.evaluate_m4_test(*args, **kwargs, expected_gradient_mode="stop_gradient")


# Kiểm tra export đúng SG rồi đối chiếu hard concepts và GT với manifest.
def validate_m4_sg_test_export(export, manifest_path):
    validate_m4_sg_config(export.get("hyperparameters"))
    return common.validate_m4_test_export(export, manifest_path)


# Gọi frozen intervention chung, yêu cầu checkpoint/export cùng variant SG.
def run_m4_sg_intervention(*args, **kwargs):
    return common.run_m4_intervention(*args, **kwargs, expected_gradient_mode="stop_gradient")


# Chuyển tham số CLI sang m4_common.run_cli cho SG; logic train/test nằm ở file dùng chung.
def main(argv=None):
    return common.run_cli("stop_gradient", argv)


if __name__ == "__main__":
    main()
