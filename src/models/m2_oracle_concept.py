"""
M2 - Oracle Concept Model for the ECBM Thesis.

Ground-truth 7-point concept labels (28-dim one-hot, same layout as Derm7ptDataset)
-> Diagnosis (Melanoma / Non-Melanoma), via Logistic Regression.

Purpose: measure the diagnostic ceiling of the 7-point concept vocabulary itself,
independent of how well any image encoder can predict those concepts. This is the
upper bound that M3 (Soft Joint CBM) and M5 (Categorical ECBM) are compared against.

Run: python src/models/m2_oracle_concept.py
"""

import json
import os
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# NOTE: src.dataset (torch) must be imported before scikit-learn on Windows, otherwise
# the two OpenMP/MKL runtimes (libiomp5md.dll) clash and the process segfaults on import.
from src.dataset import (
    CONCEPT_NAMES,
    CONCEPT_OFFSETS,
    DEFAULT_LABEL_MAPPING,
    TOTAL_CONCEPT_STATES,
)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
MAPPING_PATH = os.path.join(PROJECT_ROOT, "data", "label_mapping.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "m2_oracle_concept")

C_GRID: Tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def load_label_mapping(mapping_path: str = MAPPING_PATH) -> Dict[str, Dict[str, int]]:
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            return json.load(f)
    return DEFAULT_LABEL_MAPPING


def build_feature_names(label_mapping: Dict[str, Dict[str, int]]) -> List[str]:
    """Human-readable name for each of the 28 one-hot columns, e.g. 'streaks=irregular'."""
    names: List[str] = [""] * TOTAL_CONCEPT_STATES
    for c_name in CONCEPT_NAMES:
        idx_to_str = {v: k for k, v in label_mapping[c_name].items()}
        for c_idx, c_str in idx_to_str.items():
            names[CONCEPT_OFFSETS[c_name] + c_idx] = f"{c_name}={c_str}"
    return names


def encode_concepts(df: pd.DataFrame, label_mapping: Dict[str, Dict[str, int]]) -> np.ndarray:
    """Ground-truth 7 concept columns -> 28-dim one-hot matrix (same layout as Derm7ptDataset)."""
    n = len(df)
    X = np.zeros((n, TOTAL_CONCEPT_STATES), dtype=np.float32)
    for row_i, row in enumerate(df.itertuples(index=False)):
        row_dict = row._asdict()
        for c_name in CONCEPT_NAMES:
            c_idx = label_mapping[c_name][row_dict[c_name]]
            X[row_i, CONCEPT_OFFSETS[c_name] + c_idx] = 1.0
    return X


def load_splits(
    manifest_path: str = MANIFEST_PATH,
    label_mapping: Dict[str, Dict[str, int]] = None,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Dict[str, int]]]:
    if label_mapping is None:
        label_mapping = load_label_mapping()
    df = pd.read_csv(manifest_path)

    splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for split in ["train", "valid", "test"]:
        split_df = df[df["split"] == split].reset_index(drop=True)
        X = encode_concepts(split_df, label_mapping)
        y = split_df["diagnosis_binary"].to_numpy(dtype=np.int64)
        splits[split] = (X, y)
    return splits, label_mapping


def evaluate(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> Dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "precision_melanoma": float(precision_score(y, y_pred, pos_label=1, zero_division=0)),
        "recall_melanoma": float(recall_score(y, y_pred, pos_label=1, zero_division=0)),
        "f1_melanoma": float(f1_score(y, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def select_best_C(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    c_grid: Tuple[float, ...] = C_GRID,
) -> float:
    """Pick the L2 regularization strength that maximizes balanced accuracy on the
    official validation split (train itself is never used for model selection)."""
    best_c, best_score = c_grid[0], -1.0
    for c in c_grid:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=42)
        model.fit(X_train, y_train)
        score = balanced_accuracy_score(y_valid, model.predict(X_valid))
        print(f"    C={c:<6} -> valid balanced_accuracy={score:.4f}")
        if score > best_score:
            best_c, best_score = c, score
    return best_c


def run() -> Tuple[LogisticRegression, Dict]:
    print("=" * 70)
    print("  M2 - ORACLE CONCEPT MODEL  (7 ground-truth concepts -> Diagnosis)")
    print("=" * 70)

    splits, label_mapping = load_splits()
    feature_names = build_feature_names(label_mapping)
    X_train, y_train = splits["train"]
    X_valid, y_valid = splits["valid"]
    X_test, y_test = splits["test"]

    print(f"\nTrain: {X_train.shape}  Valid: {X_valid.shape}  Test: {X_test.shape}")
    print(
        f"Melanoma rate -> train: {y_train.mean():.3f}  "
        f"valid: {y_valid.mean():.3f}  test: {y_test.mean():.3f}"
    )

    print("\n[1] Selecting regularization strength C on the validation split:")
    best_c = select_best_C(X_train, y_train, X_valid, y_valid)
    print(f"    -> best C = {best_c}")

    print("\n[2] Fitting final model on train split, evaluating on train/valid/test:")
    model = LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000, random_state=42)
    model.fit(X_train, y_train)

    results: Dict = {"best_C": best_c}
    for split_name, (X, y) in splits.items():
        results[split_name] = evaluate(model, X, y)
        m = results[split_name]
        print(f"\n  [{split_name.upper()}]")
        print(f"    Accuracy:          {m['accuracy']:.4f}")
        print(f"    Balanced Accuracy: {m['balanced_accuracy']:.4f}")
        print(f"    Precision (mel):   {m['precision_melanoma']:.4f}")
        print(f"    Recall (mel):      {m['recall_melanoma']:.4f}")
        print(f"    F1 (mel):          {m['f1_melanoma']:.4f}")
        print(f"    ROC-AUC:           {m['roc_auc']:.4f}")
        print(f"    Confusion Matrix:  {m['confusion_matrix']}")

    print("\n[3] Concept-states most predictive of MELANOMA (largest positive LR weight):")
    coefs = model.coef_[0]
    order = np.argsort(-coefs)
    for i in order[:10]:
        print(f"    {feature_names[i]:<45} weight={coefs[i]:+.3f}")
    print("\n    Concept-states most predictive of NON-MELANOMA (largest negative LR weight):")
    for i in order[-10:][::-1]:
        print(f"    {feature_names[i]:<45} weight={coefs[i]:+.3f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    coef_df = pd.DataFrame({"concept_state": feature_names, "weight": coefs})
    coef_df = coef_df.sort_values("weight", ascending=False)
    coef_csv_path = os.path.join(OUTPUT_DIR, "concept_weights.csv")
    coef_df.to_csv(coef_csv_path, index=False)

    print(f"\n[SUCCESS] Metrics saved to:         {metrics_path}")
    print(f"[SUCCESS] Concept weights saved to: {coef_csv_path}")
    print("=" * 70)

    return model, results


if __name__ == "__main__":
    run()
