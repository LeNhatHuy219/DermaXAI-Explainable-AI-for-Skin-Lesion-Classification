import collections
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.dataset import (
    CONCEPT_NAMES,
    CONCEPT_NUM_CLASSES,
    CONCEPT_OFFSETS,
    TOTAL_CONCEPT_STATES,
    Derm7ptDataset,
    compute_diagnosis_weights,
    get_dataloaders,
)
from src.transforms import LetterboxResize, get_transforms

MANIFEST = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
MAPPING_FILE = os.path.join(PROJECT_ROOT, "data", "label_mapping.json")
passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}  -- {detail}")
        failed += 1


# --- SECTION 1: Manifest Integrity ---
print("\n[1] MANIFEST INTEGRITY")
df = pd.read_csv(MANIFEST)

check("Total rows == 1011", len(df) == 1011, f"Got {len(df)}")
check("No duplicate rows", df.duplicated().sum() == 0)
check("No duplicate case_num", df["case_num"].nunique() == len(df),
      f"{len(df) - df['case_num'].nunique()} duplicates")

# Check official split counts
split_counts = df["split"].value_counts().to_dict()
check("Train == 413", split_counts.get("train", 0) == 413)
check("Valid == 203", split_counts.get("valid", 0) == 203)
check("Test == 395", split_counts.get("test", 0) == 395)

# Check no data leakage: case_nums don't overlap across splits
for s1, s2 in [("train", "valid"), ("train", "test"), ("valid", "test")]:
    cases_1 = set(df[df["split"] == s1]["case_num"])
    cases_2 = set(df[df["split"] == s2]["case_num"])
    overlap = cases_1 & cases_2
    check(f"No case_num overlap {s1} <-> {s2}", len(overlap) == 0,
          f"Overlap: {overlap}")

# Check all concept columns exist
for c in CONCEPT_NAMES:
    check(f"Column '{c}' exists", c in df.columns)

# Check no NaN in critical columns
critical_cols = ["diagnosis_binary", "derm_path_resolved"] + CONCEPT_NAMES
for col in critical_cols:
    n_null = df[col].isnull().sum()
    check(f"No NaN in '{col}'", n_null == 0, f"{n_null} nulls")

# --- SECTION 2: Label Mapping Consistency ---
print("\n[2] LABEL MAPPING CONSISTENCY")
with open(MAPPING_FILE) as f:
    mapping = json.load(f)

check("Mapping has all 7 concepts", set(mapping.keys()) == set(CONCEPT_NAMES))

for c_name in CONCEPT_NAMES:
    # Every value in manifest must exist in mapping
    manifest_vals = set(df[c_name].unique())
    mapping_vals = set(mapping[c_name].keys())
    missing = manifest_vals - mapping_vals
    check(f"  {c_name}: all manifest values mapped",
          len(missing) == 0, f"Missing: {missing}")

    # Number of states matches CONCEPT_NUM_CLASSES
    check(f"  {c_name}: num_states == {CONCEPT_NUM_CLASSES[c_name]}",
          len(mapping[c_name]) == CONCEPT_NUM_CLASSES[c_name],
          f"Got {len(mapping[c_name])}")

    # Indices are contiguous 0..N-1
    indices = sorted(mapping[c_name].values())
    expected = list(range(CONCEPT_NUM_CLASSES[c_name]))
    check(f"  {c_name}: indices contiguous 0..{CONCEPT_NUM_CLASSES[c_name]-1}",
          indices == expected, f"Got {indices}")

# --- SECTION 3: One-Hot Encoding Correctness ---
print("\n[3] ONE-HOT ENCODING CORRECTNESS")

# Check offsets sum correctly
expected_offsets = {"pigment_network": 0, "streaks": 3, "pigmentation": 6,
                    "regression_structures": 11, "dots_and_globules": 15,
                    "blue_whitish_veil": 18, "vascular_structures": 20}
check("CONCEPT_OFFSETS correct", CONCEPT_OFFSETS == expected_offsets,
      f"Got {CONCEPT_OFFSETS}")
check("TOTAL_CONCEPT_STATES == 28", TOTAL_CONCEPT_STATES == 28)

# Manually verify one-hot for several samples
eval_tf = get_transforms(split="valid", target_size=224)
ds_all = Derm7ptDataset(MANIFEST, PROJECT_ROOT, split=None, transform=eval_tf)

onehot_sums_ok = True
mutual_exclusion_ok = True
for idx in [0, 100, 500, 1000, len(ds_all) - 1]:
    sample = ds_all[idx]
    oh = sample["concept_onehot"]

    # Sum must be exactly 7 (one state per concept group)
    if oh.sum().item() != 7.0:
        onehot_sums_ok = False

    # Within each group, exactly one bit is set
    for c_name in CONCEPT_NAMES:
        start = CONCEPT_OFFSETS[c_name]
        end = start + CONCEPT_NUM_CLASSES[c_name]
        group_sum = oh[start:end].sum().item()
        if group_sum != 1.0:
            mutual_exclusion_ok = False

check("One-hot sum == 7 for sampled rows", onehot_sums_ok)
check("Mutual exclusion within each concept group", mutual_exclusion_ok)

# Cross-check: concept_indices and concept_onehot are consistent
consistency_ok = True
for idx in range(min(50, len(ds_all))):
    sample = ds_all[idx]
    oh = sample["concept_onehot"]
    ci = sample["concept_indices"]
    for k, c_name in enumerate(CONCEPT_NAMES):
        expected_global = CONCEPT_OFFSETS[c_name] + ci[k].item()
        if oh[expected_global].item() != 1.0:
            consistency_ok = False
            break
check("concept_indices <-> concept_onehot consistency (50 samples)", consistency_ok)

# --- SECTION 4: Image Loading & Transform ---
print("\n[4] IMAGE LOADING & TRANSFORMS")

# Check all images exist
missing_imgs = []
for _, row in df.iterrows():
    p = os.path.join(PROJECT_ROOT, row["derm_path_resolved"])
    if not os.path.exists(p):
        missing_imgs.append(p)
check("All 1011 images exist on disk", len(missing_imgs) == 0,
      f"{len(missing_imgs)} missing")

# LetterboxResize preserves aspect ratio
test_img = Image.new("RGB", (768, 512), (128, 128, 128))
lb = LetterboxResize((224, 224))
result = lb(test_img)
check("LetterboxResize output is 224x224", result.size == (224, 224))

# With 768x512 (3:2), scale = 224/768 = 0.2917
# new_w = 224, new_h = round(512 * 0.2917) = 149
# So padding should exist on top/bottom but not left/right
arr = np.array(result)
# Top row should be padding (black)
top_is_pad = np.all(arr[0, :, :] == 0)
# Middle should have content
mid_row = arr[112, 112, :]
mid_has_content = np.any(mid_row > 0)
check("LetterboxResize: padding at top (aspect ratio preserved)",
      top_is_pad and mid_has_content)

# Check train vs eval transforms produce different results (stochastic vs deterministic)
train_tf = get_transforms("train", 224, augment=True)
eval_tf = get_transforms("valid", 224, augment=False)

real_img_path = os.path.join(PROJECT_ROOT, df.iloc[0]["derm_path_resolved"])
real_img = Image.open(real_img_path).convert("RGB")

eval_t1 = eval_tf(real_img)
eval_t2 = eval_tf(real_img)
check("Eval transform is deterministic", torch.equal(eval_t1, eval_t2))

# Train transform should sometimes differ (run 10 times)
train_results = [train_tf(real_img) for _ in range(10)]
all_same = all(torch.equal(train_results[0], t) for t in train_results[1:])
check("Train transform is stochastic (augmentation)", not all_same)

# --- SECTION 5: Class Weights ---
print("\n[5] CLASS WEIGHTS & IMBALANCE HANDLING")

weights = compute_diagnosis_weights(MANIFEST, PROJECT_ROOT)
train_df = df[df["split"] == "train"]
n_0 = (train_df["diagnosis_binary"] == 0).sum()
n_1 = (train_df["diagnosis_binary"] == 1).sum()
n_total = len(train_df)

expected_w0 = n_total / (2.0 * n_0)
expected_w1 = n_total / (2.0 * n_1)
check(f"w0 = {expected_w0:.4f} (Non-Melanoma)", abs(weights[0].item() - expected_w0) < 1e-5)
check(f"w1 = {expected_w1:.4f} (Melanoma)", abs(weights[1].item() - expected_w1) < 1e-5)
check("Melanoma weight > Non-Melanoma weight (minority upweighted)",
      weights[1] > weights[0])
check("Weights computed ONLY from train split",
      n_total == 413, f"Used {n_total} samples")

# --- SECTION 6: DataLoader Batch Structure ---
print("\n[6] DATALOADER BATCH STRUCTURE")

loaders = get_dataloaders(MANIFEST, PROJECT_ROOT, batch_size=8, num_workers=0,
                          target_size=224, augment_train=False)

for split_name in ["train", "valid", "test"]:
    batch = next(iter(loaders[split_name]))
    bs = batch["image"].shape[0]
    check(f"{split_name} batch image shape: ({bs}, 3, 224, 224)",
          batch["image"].shape == (bs, 3, 224, 224))
    check(f"{split_name} batch label shape: ({bs},)",
          batch["label"].shape == (bs,))
    check(f"{split_name} batch concept_onehot shape: ({bs}, 28)",
          batch["concept_onehot"].shape == (bs, 28))
    check(f"{split_name} batch concept_indices shape: ({bs}, 7)",
          batch["concept_indices"].shape == (bs, 7))

    # concept_labels is a dict of 7 keys, each value shape (bs,)
    cl = batch["concept_labels"]
    check(f"{split_name} concept_labels has 7 keys",
          set(cl.keys()) == set(CONCEPT_NAMES))
    for c_name in CONCEPT_NAMES:
        in_range = (cl[c_name] >= 0).all() and (cl[c_name] < CONCEPT_NUM_CLASSES[c_name]).all()
        if not in_range:
            check(f"{split_name} {c_name} values in range", False,
                  f"Values: {cl[c_name].tolist()}")
            break
    else:
        check(f"{split_name} all concept_labels values in valid range", True)

# --- SECTION 7: Rare State Analysis ---
print("\n[7] RARE CONCEPT STATE ANALYSIS (for reporting)")

for c_name in CONCEPT_NAMES:
    for state_name, state_idx in mapping[c_name].items():
        train_count = (train_df[c_name] == state_name).sum()
        test_count = (df[df["split"] == "test"][c_name] == state_name).sum()
        if train_count <= 5 or test_count == 0:
            print(f"  [WARN] {c_name}={state_name}: train={train_count}, test={test_count}")

# --- SECTION 8: Concept Profile Inconsistency ---
print("\n[8] CONCEPT INCONSISTENCY VERIFICATION")

n_inconsistent = df["is_inconsistent_profile"].sum()
print(f"  Total inconsistent profile samples: {n_inconsistent} / {len(df)} "
      f"({100*n_inconsistent/len(df):.1f}%)")

# Verify inconsistency flag by recomputing
profiles = df.groupby("concept_profile")["diagnosis_binary"].nunique()
inconsistent_profiles = set(profiles[profiles > 1].index)
recomputed_flags = df["concept_profile"].isin(inconsistent_profiles)
mismatch = (recomputed_flags != df["is_inconsistent_profile"]).sum()
check("is_inconsistent_profile recomputation matches", mismatch == 0,
      f"{mismatch} mismatches")

# Summary
total = passed + failed
print(f"\nAudit complete: {passed}/{total} checks passed, {failed} failed.")

