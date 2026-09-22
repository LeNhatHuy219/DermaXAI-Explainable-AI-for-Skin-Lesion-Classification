import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from src.dataset import (
    CONCEPT_NAMES,
    CONCEPT_NUM_CLASSES,
    TOTAL_CONCEPT_STATES,
    Derm7ptDataset,
    compute_diagnosis_weights,
    get_dataloaders,
)


def run_pipeline_tests():
    manifest_path = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
    assert os.path.exists(manifest_path), f"Không tìm thấy file manifest tại {manifest_path}"

    # 1. Kiểm tra cấu hình và số lượng khái niệm (7 nhóm, 28 trạng thái)
    assert len(CONCEPT_NAMES) == 7
    assert TOTAL_CONCEPT_STATES == 28

    # 2. Kiểm tra số lượng mẫu trong 3 tập train (413), valid (203), test (395)
    dataloaders_dict = get_dataloaders(
        manifest_path=manifest_path,
        project_root=PROJECT_ROOT,
        batch_size=16,
        num_workers=0,
        target_size=224,
        augment_train=True,
    )

    datasets = dataloaders_dict["datasets"]
    train_len = len(datasets["train"])
    val_len = len(datasets["valid"])
    test_len = len(datasets["test"])

    assert train_len == 413, f"Số mẫu train {train_len} != 413"
    assert val_len == 203, f"Số mẫu valid {val_len} != 203"
    assert test_len == 395, f"Số mẫu test {test_len} != 395"

    # 3. Kiểm tra cấu trúc một mẫu dữ liệu (ảnh 224x224, vector one-hot 28 chiều)
    sample = datasets["train"][0]
    img = sample["image"]
    concept_onehot = sample["concept_onehot"]
    concept_indices = sample["concept_indices"]

    assert img.shape == (3, 224, 224)
    assert concept_onehot.shape == (28,)
    assert concept_onehot.sum().item() == 7.0
    assert concept_indices.shape == (7,)

    # 4. Kiểm tra trọng số bù trừ mất cân bằng lớp trên tập train
    weights = dataloaders_dict["class_weights"]
    expected_w0 = 413.0 / (2.0 * 323.0)
    expected_w1 = 413.0 / (2.0 * 90.0)
    assert abs(weights[0].item() - expected_w0) < 1e-4
    assert abs(weights[1].item() - expected_w1) < 1e-4

    # 5. Kiểm tra gom nhóm batch trong DataLoader
    train_loader = dataloaders_dict["train"]
    batch = next(iter(train_loader))
    assert batch["image"].shape == (16, 3, 224, 224)
    assert batch["label"].shape == (16,)
    assert batch["concept_onehot"].shape == (16, 28)
    assert batch["concept_indices"].shape == (16, 7)

    print("Tất cả các kiểm tra pipeline dữ liệu đã vượt qua thành công.")


if __name__ == "__main__":
    run_pipeline_tests()
