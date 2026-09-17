import os
import sys

# Add PROJECT directory to sys.path so src can be imported directly
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
    print("=" * 60)
    print("       DERM7PT DATA PIPELINE VERIFICATION SUITE       ")
    print("=" * 60)

    manifest_path = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
    assert os.path.exists(manifest_path), f"Manifest not found at {manifest_path}"

    # Test 1: Concept configuration
    print("\n[1] Checking Concept Definitions:")
    print(f"  - Total concept groups: {len(CONCEPT_NAMES)}")
    print(f"  - Total discrete states: {TOTAL_CONCEPT_STATES} (Expected: 28)")
    assert TOTAL_CONCEPT_STATES == 28, "Total concept states must be 28!"
    print("  -> PASSED: Concept dimensions verified.")

    # Test 2: Dataset Instantiation & Split Sizes
    print("\n[2] Checking Dataset Splits & Counts:")
    dataloaders_dict = get_dataloaders(
        manifest_path=manifest_path,
        project_root=PROJECT_ROOT,
        batch_size=16,
        num_workers=0,  # Single-process for test
        target_size=224,
        augment_train=True,
    )

    datasets = dataloaders_dict["datasets"]
    train_len = len(datasets["train"])
    val_len = len(datasets["valid"])
    test_len = len(datasets["test"])
    total_len = train_len + val_len + test_len

    print(f"  - Train count: {train_len} (Expected: 413)")
    print(f"  - Valid count: {val_len} (Expected: 203)")
    print(f"  - Test count:  {test_len} (Expected: 395)")
    print(f"  - Total count: {total_len} (Expected: 1011)")

    assert train_len == 413, f"Train count {train_len} != 413"
    assert val_len == 203, f"Valid count {val_len} != 203"
    assert test_len == 395, f"Test count {test_len} != 395"
    assert total_len == 1011, f"Total count {total_len} != 1011"
    print("  -> PASSED: Dataset splits match official Derm7pt partitioning.")

    # Test 3: Single Sample Verification
    print("\n[3] Checking Single Sample Integrity (Index 0):")
    sample = datasets["train"][0]

    img = sample["image"]
    label = sample["label"]
    concept_onehot = sample["concept_onehot"]
    concept_indices = sample["concept_indices"]
    meta = sample["meta"]

    print(f"  - Image tensor shape: {tuple(img.shape)} (Expected: (3, 224, 224))")
    print(f"  - Image value range: [{img.min().item():.3f}, {img.max().item():.3f}]")
    print(f"  - Diagnosis label: {label.item()} (Diagnosis: {meta['diagnosis']})")
    print(f"  - Concept indices shape: {tuple(concept_indices.shape)} (Values: {concept_indices.tolist()})")
    print(f"  - Concept one-hot shape: {tuple(concept_onehot.shape)} (Sum: {concept_onehot.sum().item():.1f})")
    print(f"  - Case num: {meta['case_num']}, Inconsistent: {meta['is_inconsistent']}")

    assert img.shape == (3, 224, 224), "Image shape mismatch!"
    assert concept_onehot.shape == (28,), "Concept one-hot vector must have length 28!"
    assert concept_onehot.sum().item() == 7.0, "Each sample must have exactly 7 active concept states!"
    assert concept_indices.shape == (7,), "Concept indices vector must have length 7!"
    print("  -> PASSED: Sample structure and one-hot encoding verified.")

    # Test 4: Class Weights Calculation
    print("\n[4] Checking Training Class Weights:")
    weights = dataloaders_dict["class_weights"]
    print(f"  - Non-Melanoma weight: {weights[0].item():.4f}")
    print(f"  - Melanoma weight:     {weights[1].item():.4f}")
    # Train set: Non-Melanoma = 323, Melanoma = 90
    expected_w0 = 413.0 / (2.0 * 323.0)
    expected_w1 = 413.0 / (2.0 * 90.0)
    assert abs(weights[0].item() - expected_w0) < 1e-4
    assert abs(weights[1].item() - expected_w1) < 1e-4
    print(f"  -> PASSED: Loss weights correctly calculated strictly on train set.")

    # Test 5: Batch Collation in DataLoader
    print("\n[5] Checking DataLoader Batch Collation:")
    train_loader = dataloaders_dict["train"]
    batch = next(iter(train_loader))

    batch_imgs = batch["image"]
    batch_labels = batch["label"]
    batch_onehot = batch["concept_onehot"]
    batch_indices = batch["concept_indices"]

    print(f"  - Batch images:         {tuple(batch_imgs.shape)}")
    print(f"  - Batch labels:         {tuple(batch_labels.shape)}")
    print(f"  - Batch concept onehot: {tuple(batch_onehot.shape)}")
    print(f"  - Batch concept indices:{tuple(batch_indices.shape)}")

    assert batch_imgs.shape == (16, 3, 224, 224)
    assert batch_labels.shape == (16,)
    assert batch_onehot.shape == (16, 28)
    assert batch_indices.shape == (16, 7)
    print("  -> PASSED: Batch collation operates smoothly.")

    print("\n" + "=" * 60)
    print(" [SUCCESS] ALL 5 DATA PIPELINE CHECKS PASSED PERFECTLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline_tests()
