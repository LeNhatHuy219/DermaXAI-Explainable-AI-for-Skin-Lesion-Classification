"""Experiment outputs and frozen data/schema metadata shared by the runners."""
import hashlib
import json
from pathlib import Path

from .dataset import CONCEPT_NAMES, CONCEPT_NUM_CLASSES


def prepare_training_outputs(default_paths, checkpoint_path=None, results_path=None,
                             overwrite=False, save_results=True):
    """Resolve distinct outputs and reject existing artifacts before training starts."""
    checkpoint = Path(checkpoint_path or default_paths[0]).resolve()
    results = Path(results_path or default_paths[1]).resolve()
    paths = [checkpoint, results]
    if len(set(paths)) != 2:
        raise ValueError("Checkpoint and JSON results must use distinct paths")
    for path in paths if save_results else [checkpoint]:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Artifact already exists: {path}. Choose another path or pass --overwrite.")
    return tuple(str(path) for path in paths)


def save_experiment_results(path, validation=None, test=None, overwrite=False):
    """Save both evaluation splits in one JSON; keep validation when re-exporting test."""
    path = Path(path)
    checkpoint_path = (test or validation or {}).get("checkpoint_path")
    if checkpoint_path and Path(checkpoint_path).resolve() == path.resolve():
        raise ValueError("Checkpoint and JSON results must use distinct paths")
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Artifact already exists: {path}. Choose another path or pass --overwrite.")
        if validation is None and test is not None:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if "validation_metrics" in existing:
                validation = existing
    validation = validation or {}
    test = test or {}
    if validation and test:
        for key in ("model", "checkpoint_path", "best_epoch", "hyperparameters",
                    "decision_threshold", "concept_schema"):
            left_value, right_value = validation.get(key), test.get(key)
            if key == "checkpoint_path" and left_value and right_value:
                left_value, right_value = Path(left_value).resolve(), Path(right_value).resolve()
            elif key == "hyperparameters":
                # JSON converts numeric C-grid dictionary keys to strings.
                left_value = json.loads(json.dumps(left_value))
                right_value = json.loads(json.dumps(right_value))
            if left_value != right_value:
                raise ValueError(f"Validation/test {key} differs; cannot combine different runs")
        left, right = validation.get("manifest_fingerprint"), test.get("manifest_fingerprint")
        if left and right:
            if left["normalized_sha256"] != right["normalized_sha256"]:
                raise ValueError("Validation/test manifests differ")
        elif validation.get("manifest_sha256") != test.get("manifest_sha256"):
            raise ValueError("Validation/test manifests differ")
    result = {**validation, **test}
    result["mode"] = ("train_validation_test" if "validation_metrics" in result and "test_metrics" in result
                      else "train_validation" if "validation_metrics" in result else "final_test")
    validation_summary = validation.get("validation_summary", validation.get("summary"))
    test_summary = test.get("test_summary", test.get("summary"))
    if validation_summary:
        result["validation_summary"] = validation_summary
    if test_summary:
        result["test_summary"] = test_summary
    if validation_summary or test_summary:
        result["summary"] = "\n\n".join(s for s in (validation_summary, test_summary) if s)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Kết quả đã lưu tại: {path.resolve()}")
    return result


def manifest_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_fingerprint(path):
    raw = Path(path).read_bytes()
    return {
        "version": "newline_lf_v1",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "normalized_sha256": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
    }


def validate_manifest(checkpoint, path):
    current = manifest_fingerprint(path)
    saved = checkpoint.get("manifest_fingerprint")
    if saved is not None:
        matches = (saved.get("version") == current["version"] and
                   saved.get("normalized_sha256") == current["normalized_sha256"])
    else:
        # Exact legacy compatibility for LF/CRLF only; content changes still fail.
        lf = Path(path).read_bytes().replace(b"\r\n", b"\n")
        hashes = {hashlib.sha256(data).hexdigest() for data in
                  (Path(path).read_bytes(), lf, lf.replace(b"\n", b"\r\n"))}
        matches = checkpoint.get("manifest_sha256") in hashes
    if not matches:
        raise ValueError("The manifest differs from the one used for training")
    return current


def _schema_hash(schema):
    payload = {k: v for k, v in schema.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def load_concept_schema(manifest_path, mapping_path=None):
    mapping_path = Path(mapping_path) if mapping_path else Path(manifest_path).resolve().parent / "label_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if set(mapping) != set(CONCEPT_NAMES):
        raise ValueError("Label mapping must define exactly the seven concept groups")
    offsets = {}
    offset = 0
    for name in CONCEPT_NAMES:
        values = list(mapping[name].values())
        count = CONCEPT_NUM_CLASSES[name]
        if (any(type(v) is not int for v in values) or
                len(values) != count or sorted(values) != list(range(count))):
            raise ValueError(f"Invalid contiguous state indices for {name}")
        offsets[name] = offset
        offset += count
    schema = {"version": 1, "concept_names": list(CONCEPT_NAMES),
              "concept_num_classes": dict(CONCEPT_NUM_CLASSES), "offsets": offsets,
              "total_states": offset, "label_mapping": mapping}
    schema["sha256"] = _schema_hash(schema)
    return schema


def validate_concept_schema(checkpoint, manifest_path):
    saved = checkpoint.get("concept_schema")
    if saved is None:
        raise ValueError("Checkpoint has no frozen concept schema; retrain with the finalized pipeline")
    if saved.get("sha256") != _schema_hash(saved):
        raise ValueError("Checkpoint concept schema is corrupt")
    current = load_concept_schema(manifest_path)
    if saved["sha256"] != current["sha256"]:
        raise ValueError("Concept schema differs from the checkpoint (mapping/order/cardinalities)")
    return saved
