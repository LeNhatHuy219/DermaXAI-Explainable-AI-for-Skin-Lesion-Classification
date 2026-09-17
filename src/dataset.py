import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from .transforms import get_transforms
except (ImportError, ValueError):
    from transforms import get_transforms

# Standard Concept Definitions for Derm7pt (7-Point Checklist)
CONCEPT_NAMES: List[str] = [
    "pigment_network",
    "streaks",
    "pigmentation",
    "regression_structures",
    "dots_and_globules",
    "blue_whitish_veil",
    "vascular_structures",
]

# Number of discrete states per concept group
CONCEPT_NUM_CLASSES: Dict[str, int] = {
    "pigment_network": 3,
    "streaks": 3,
    "pigmentation": 5,
    "regression_structures": 4,
    "dots_and_globules": 3,
    "blue_whitish_veil": 2,
    "vascular_structures": 8,
}

TOTAL_CONCEPT_STATES: int = sum(CONCEPT_NUM_CLASSES.values())  # 28 states

# Precompute starting offset for each concept in the 28-dimensional one-hot vector
CONCEPT_OFFSETS: Dict[str, int] = {}
_offset = 0
for name in CONCEPT_NAMES:
    CONCEPT_OFFSETS[name] = _offset
    _offset += CONCEPT_NUM_CLASSES[name]


# Default fallback label mapping if JSON is not passed
DEFAULT_LABEL_MAPPING: Dict[str, Dict[str, int]] = {
    "pigment_network": {"absent": 0, "atypical": 1, "typical": 2},
    "streaks": {"absent": 0, "irregular": 1, "regular": 2},
    "pigmentation": {
        "absent": 0,
        "diffuse irregular": 1,
        "diffuse regular": 2,
        "localized irregular": 3,
        "localized regular": 4,
    },
    "regression_structures": {
        "absent": 0,
        "blue areas": 1,
        "combinations": 2,
        "white areas": 3,
    },
    "dots_and_globules": {"absent": 0, "irregular": 1, "regular": 2},
    "blue_whitish_veil": {"absent": 0, "present": 1},
    "vascular_structures": {
        "absent": 0,
        "arborizing": 1,
        "comma": 2,
        "dotted": 3,
        "hairpin": 4,
        "linear irregular": 5,
        "within regression": 6,
        "wreath": 7,
    },
}


class Derm7ptDataset(Dataset):
    """
    Dataset class for Derm7pt supporting:
    - M0: Majority Baseline
    - M1: Black-box CNN/Transformer (Image -> Diagnosis)
    - M2: Oracle Concept Classifier (GT Concepts -> Diagnosis)
    - M3: Soft Joint CBM (Image -> Soft Concepts -> Diagnosis)
    - M4: Hard Sequential CBM (Image -> Hard Concepts -> Diagnosis)
    - M5: Energy-Based CBM (Image, Concepts, Diagnosis Joint Energy)
    """

    def __init__(
        self,
        manifest_path: str,
        project_root: str,
        split: Optional[Union[str, List[str]]] = None,
        transform=None,
        label_mapping: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        """
        Parameters:
        -----------
        manifest_path : str
            Path to manifest.csv.
        project_root : str
            Base directory from which derm_path_resolved is located.
        split : str or list of str, optional
            'train', 'valid' (or 'val'), 'test', or list of them. None loads all.
        transform : callable, optional
            Image transform pipeline.
        label_mapping : dict, optional
            Concept string-to-int mapping. If None, loads from label_mapping.json or defaults.
        """
        self.project_root = project_root
        self.transform = transform

        if not os.path.isabs(manifest_path):
            manifest_path = os.path.join(project_root, manifest_path)
        self.df = pd.read_csv(manifest_path)

        # Standardize split filter
        if split is not None:
            if isinstance(split, str):
                split_val = "valid" if split.lower() in ["val", "valid"] else split.lower()
                self.df = self.df[self.df["split"] == split_val].copy()
            elif isinstance(split, (list, tuple)):
                splits = ["valid" if s.lower() in ["val", "valid"] else s.lower() for s in split]
                self.df = self.df[self.df["split"].isin(splits)].copy()

        self.df = self.df.reset_index(drop=True)

        # Label mapping for concepts
        if label_mapping is not None:
            self.label_mapping = label_mapping
        else:
            # Check for label_mapping.json in same dir as manifest
            mapping_file = os.path.join(os.path.dirname(manifest_path), "label_mapping.json")
            if os.path.exists(mapping_file):
                with open(mapping_file, "r") as f:
                    self.label_mapping = json.load(f)
            else:
                self.label_mapping = DEFAULT_LABEL_MAPPING

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict, str, int]]:
        row = self.df.iloc[idx]

        # 1. Load Image
        img_rel_path = row["derm_path_resolved"]
        img_full_path = os.path.join(self.project_root, img_rel_path)
        img = Image.open(img_full_path).convert("RGB")

        if self.transform is not None:
            image_tensor = self.transform(img)
        else:
            # Fallback simple tensor conversion if no transform provided
            image_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        # 2. Diagnosis Label (Binary: 0 for Non-Melanoma, 1 for Melanoma)
        target_y = int(row["diagnosis_binary"])
        label_tensor = torch.tensor(target_y, dtype=torch.long)

        # 3. Concept Labels (per group integer class & 28-dimensional one-hot)
        concept_labels: Dict[str, torch.Tensor] = {}
        concept_indices_list: List[int] = []
        concept_onehot = torch.zeros(TOTAL_CONCEPT_STATES, dtype=torch.float32)

        for c_name in CONCEPT_NAMES:
            c_str_val = row[c_name]
            c_idx = self.label_mapping[c_name][c_str_val]
            concept_labels[c_name] = torch.tensor(c_idx, dtype=torch.long)
            concept_indices_list.append(c_idx)

            # Set 1 at corresponding global one-hot position
            global_idx = CONCEPT_OFFSETS[c_name] + c_idx
            concept_onehot[global_idx] = 1.0

        concept_indices = torch.tensor(concept_indices_list, dtype=torch.long)

        # 4. Metadata for error analysis, intervention tracking & reporting
        meta = {
            "case_num": int(row["case_num"]),
            "source_index": int(row["source_index"]),
            "split": str(row["split"]),
            "diagnosis": str(row["diagnosis"]),
            "concept_profile": str(row["concept_profile"]),
            "is_inconsistent": bool(row["is_inconsistent_profile"]),
            "derm_path": str(img_rel_path),
        }

        return {
            "image": image_tensor,
            "label": label_tensor,
            "concept_labels": concept_labels,
            "concept_indices": concept_indices,
            "concept_onehot": concept_onehot,
            "meta": meta,
        }


def compute_diagnosis_weights(manifest_path: str, project_root: str) -> torch.Tensor:
    """
    Computes inverse class frequencies on the TRAINING set only:
    w_j = N / (2 * N_j)
    Prevents data leakage from val/test.
    """
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.join(project_root, manifest_path)
    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"]

    counts = train_df["diagnosis_binary"].value_counts().to_dict()
    n_total = len(train_df)
    n_0 = counts.get(0, 1)
    n_1 = counts.get(1, 1)

    w0 = n_total / (2.0 * n_0)
    w1 = n_total / (2.0 * n_1)

    return torch.tensor([w0, w1], dtype=torch.float32)


def get_dataloaders(
    manifest_path: str,
    project_root: str,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
    target_size: int = 224,
    augment_train: bool = True,
    use_weighted_sampler: bool = False,
) -> Dict[str, Union[DataLoader, torch.Tensor, Dict[str, Derm7ptDataset]]]:
    """
    Constructs train, valid, and test DataLoaders following the official Derm7pt split.

    Returns:
    --------
    dict with keys:
        'train': DataLoader,
        'valid': DataLoader,
        'test': DataLoader,
        'class_weights': Tensor [w0, w1] for balanced cross-entropy,
        'datasets': {'train': ..., 'valid': ..., 'test': ...}
    """
    train_transform = get_transforms(split="train", target_size=target_size, augment=augment_train)
    eval_transform = get_transforms(split="valid", target_size=target_size, augment=False)

    train_dataset = Derm7ptDataset(
        manifest_path=manifest_path,
        project_root=project_root,
        split="train",
        transform=train_transform,
    )
    valid_dataset = Derm7ptDataset(
        manifest_path=manifest_path,
        project_root=project_root,
        split="valid",
        transform=eval_transform,
    )
    test_dataset = Derm7ptDataset(
        manifest_path=manifest_path,
        project_root=project_root,
        split="test",
        transform=eval_transform,
    )

    if pin_memory and torch.backends.mps.is_available():
        pin_memory = False

    class_weights = compute_diagnosis_weights(manifest_path, project_root)

    # Optional WeightedRandomSampler for training batch balancing
    train_sampler = None
    shuffle_train = True
    if use_weighted_sampler:
        targets = train_dataset.df["diagnosis_binary"].values
        sample_weights = [class_weights[t].item() for t in targets]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle_train = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
        "class_weights": class_weights,
        "datasets": {
            "train": train_dataset,
            "valid": valid_dataset,
            "test": test_dataset,
        },
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Running src/dataset.py directly as standalone script...")
    print("=" * 60)

    # Determine paths relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    manifest_path = os.path.join(project_root, "data", "manifest.csv")

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        exit(1)

    print(f"Loading data from: {manifest_path}")
    bundle = get_dataloaders(
        manifest_path=manifest_path,
        project_root=project_root,
        batch_size=8,
        num_workers=0,
        target_size=224,
    )

    print(f"Train samples: {len(bundle['datasets']['train'])}")
    print(f"Valid samples: {len(bundle['datasets']['valid'])}")
    print(f"Test samples:  {len(bundle['datasets']['test'])}")
    print(f"Class weights: {bundle['class_weights'].tolist()}")

    batch = next(iter(bundle["train"]))
    print(f"Sample Batch Images shape:  {tuple(batch['image'].shape)}")
    print(f"Sample Batch Labels shape:  {tuple(batch['label'].shape)}")
    print(f"Sample Batch Concept shape: {tuple(batch['concept_onehot'].shape)}")
    print("[SUCCESS] src/dataset.py executed standalone successfully!")

