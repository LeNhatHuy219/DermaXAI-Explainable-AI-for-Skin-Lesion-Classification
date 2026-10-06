# Logic dùng chung cho M4-ST và M4-SG: train, validation, frozen test và intervention.
# Đầu vào: manifest/ảnh khi train/test; checkpoint + JSON predictions khi intervention.
# ST: diagnosis loss truyền gradient qua xấp xỉ straight-through về concept/backbone.
# SG: diagnosis loss cập nhật head; concept/backbone học từ concept loss.
# Cả hai đều đưa predicted hard one-hot 28 chiều vào diagnosis head ở forward.
# Đầu ra JSON: results/<metric>/<m4_st,m4_sg>/<cấu hình>/seed<seed>.json.
# BAcc cũ: checkpoints/bacc/baseline/ và results/bacc/; đường dẫn riêng được ưu tiên.
# Đọc run_m4_experiment -> evaluate_m4_test -> run_m4_intervention; run_cli nối các bước.
# File này được runner ST/SG gọi, không phải launcher train độc lập.

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load torch through the data module before sklearn (Windows DLL import order).
from src.dataset import (CONCEPT_NAMES, CONCEPT_NUM_CLASSES, Derm7ptDataset,
                         compute_concept_statistics, get_dataloaders)
from experiments import run_m3 as shared
import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

from src.intervention import evaluate_hard_interventions, validate_hard_concepts
from src.metrics import (compute_concept_metrics, compute_metrics,
                         print_concept_metrics_table, print_metrics_table)
from src.cbm import get_hard_joint_cbm, get_hard_stop_gradient_cbm, make_diagnosis_head
from src.protocol import (load_concept_schema, manifest_fingerprint, prepare_training_outputs,
                          save_experiment_results, validate_concept_schema, validate_manifest)
from src.transforms import get_transforms
from src.selection import (DEFAULT_CHECKPOINT_METRIC, selection_name, selection_score,
                           selection_suffix, metric_from_config, add_selection_argument,
                           checkpoint_directory, configuration_name, results_run_path)

M4_MODEL = "M4_HardJointCBM"
M4_PROTOCOL = "hard_joint_st_state_weighted_v1"
M4_SG_MODEL = "M4_HardStopGradientCBM"
M4_SG_PROTOCOL = "hard_joint_sg_state_weighted_v1"
DIAGNOSIS_HEADS = {"linear": "Linear(28, 2)", "mlp128": "MLP(28, 128, 2)"}
M4_VARIANTS = {
    "straight_through": {"model": M4_MODEL, "protocol": M4_PROTOCOL,
                         "gradient_estimator": "deterministic_softmax_straight_through"},
    "stop_gradient": {"model": M4_SG_MODEL, "protocol": M4_SG_PROTOCOL,
                      "gradient_estimator": "stop_gradient"},
}
DEFAULT_EPOCHS = shared.DEFAULT_EPOCHS
DEFAULT_BATCH_SIZE = shared.DEFAULT_BATCH_SIZE
DEFAULT_LR = shared.DEFAULT_LR
DEFAULT_WEIGHT_DECAY = shared.DEFAULT_WEIGHT_DECAY
DEFAULT_SEED = shared.DEFAULT_SEED
DEFAULT_AUGMENTATION_PRESET = shared.DEFAULT_AUGMENTATION_PRESET
DEFAULT_CONCEPT_LOSS_WEIGHT = shared.DEFAULT_CONCEPT_LOSS_WEIGHT


# Tạo checkpoint/JSON paths theo cấu hình, seed và tiêu chí chọn checkpoint.
def _default_paths(seed=DEFAULT_SEED, augmentation_preset=DEFAULT_AUGMENTATION_PRESET,
                   concept_loss_weight=DEFAULT_CONCEPT_LOSS_WEIGHT, gradient_mode="straight_through",
                   diagnosis_lr=None, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR,
                   weight_decay=DEFAULT_WEIGHT_DECAY, diagnosis_head="linear",
                   checkpoint_metric=DEFAULT_CHECKPOINT_METRIC):
    _variant_metadata(gradient_mode)
    stem = "m4_hard_sg_cbm" if gradient_mode == "stop_gradient" else "m4_hard_joint_cbm"
    if diagnosis_head not in DIAGNOSIS_HEADS:
        raise ValueError("Unknown diagnosis head")
    if diagnosis_head != "linear":
        stem += f"_{diagnosis_head}"
    if augmentation_preset != DEFAULT_AUGMENTATION_PRESET:
        stem += f"_{augmentation_preset}"
    if concept_loss_weight != DEFAULT_CONCEPT_LOSS_WEIGHT:
        stem += f"_lambda{concept_loss_weight:g}"
    # Cấu hình tuning có tên riêng để không ghi đè checkpoint/kết quả baseline.
    if diagnosis_lr is not None:
        stem += f"_headlr{diagnosis_lr:g}"
    if diagnosis_lr is not None or diagnosis_head != "linear":
        if epochs != DEFAULT_EPOCHS:
            stem += f"_epochs{epochs}"
        if lr != DEFAULT_LR:
            stem += f"_lr{lr:g}"
        if weight_decay != DEFAULT_WEIGHT_DECAY:
            stem += f"_wd{weight_decay:g}"
    stem += f"_seed{seed}"
    stem += selection_suffix(checkpoint_metric)
    model = "m4_sg" if gradient_mode == "stop_gradient" else "m4_st"
    configuration = configuration_name(diagnosis_head, epochs, diagnosis_lr,
        augmentation_preset=augmentation_preset, concept_loss_weight=concept_loss_weight,
        lr=lr, weight_decay=weight_decay)
    return (str(checkpoint_directory(ROOT, checkpoint_metric) / f"{stem}_best.pth"),
            str(results_run_path(ROOT, checkpoint_metric, model, configuration, seed)))


# Kiểm tra gradient mode và lấy model/protocol metadata đúng cho ST hoặc SG.
def _variant_metadata(gradient_mode):
    if gradient_mode not in M4_VARIANTS:
        raise ValueError("Unknown M4 gradient mode")
    return M4_VARIANTS[gradient_mode]


# Nhận diện ST/SG qua metadata checkpoint; không suy đoán từ tên file.
def _gradient_mode_from_config(config):
    if isinstance(config, dict):
        for mode, metadata in M4_VARIANTS.items():
            if all(config.get(key) == value for key, value in metadata.items()):
                return mode
    raise ValueError("Checkpoint is not the finalized M4 hard/ST protocol or M4 hard/SG protocol")


# Chặn ghép checkpoint của biến thể khác với runner đang gọi.
def _require_gradient_mode(config, expected_gradient_mode):
    if expected_gradient_mode is not None:
        _variant_metadata(expected_gradient_mode)
        if _gradient_mode_from_config(config) != expected_gradient_mode:
            protocol = "hard/SG" if expected_gradient_mode == "stop_gradient" else "hard/ST"
            raise ValueError(f"Checkpoint gradient mode differs; expected M4 {protocol} protocol")


# Dựng HardJointCBM hoặc HardStopGradientCBM, với diagnosis head đúng preset.
def _build_model(gradient_mode, concept_names, concept_num_classes, pretrained, dropout=0.2,
                 diagnosis_head="linear"):
    _variant_metadata(gradient_mode)
    factory = get_hard_stop_gradient_cbm if gradient_mode == "stop_gradient" else get_hard_joint_cbm
    return factory(concept_names, concept_num_classes, pretrained=pretrained, dropout=dropout,
                   diagnosis_head=diagnosis_head)


# Đọc head Linear/MLP128 từ config để test/intervention dựng đúng kiến trúc.
def _head_preset(config):
    # Chỉ chấp nhận hai kiến trúc đã khai báo; checkpoint cũ giữ head Linear.
    for preset, description in DIAGNOSIS_HEADS.items():
        if config.get("diagnosis_head") == description:
            return preset
    raise ValueError("Checkpoint is not the finalized M4 hard/ST protocol or M4 hard/SG protocol: unknown diagnosis head")


# Tạo AdamW; LR riêng tách parameter groups, còn đường gradient do ST/SG quyết định.
def _build_optimizer(model, lr, weight_decay, diagnosis_lr=None):
    # Không truyền diagnosis_lr thì giữ nguyên optimizer của các runs cũ.
    if diagnosis_lr is None:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if not np.isfinite(diagnosis_lr) or diagnosis_lr <= 0:
        raise ValueError("diagnosis learning rate must be positive and finite")
    # Head khởi tạo ngẫu nhiên có thể cần LR lớn hơn backbone pretrained.
    # Tách theo identity tham số; mỗi parameter xuất hiện đúng một lần.
    diagnosis_parameters = list(model.diagnosis_head.parameters())
    diagnosis_ids = {id(parameter) for parameter in diagnosis_parameters}
    concept_parameters = [parameter for parameter in model.parameters()
                          if id(parameter) not in diagnosis_ids]
    return torch.optim.AdamW([
        {"params": concept_parameters, "lr": lr, "name": "concept_predictor"},
        {"params": diagnosis_parameters, "lr": diagnosis_lr, "name": "diagnosis_head"},
    ], weight_decay=weight_decay)


# Lấy đường dẫn đúng từ config checkpoint: variant/head/LR/epochs/seed/metric.
def _config_paths(config, gradient_mode):
    # Test/intervention suy ra đúng đường dẫn từ cấu hình checkpoint đã đóng băng.
    return _default_paths(config["seed"], config["augmentation_preset"],
        config["concept_loss_weight"], gradient_mode,
        diagnosis_lr=config.get("diagnosis_learning_rate"), epochs=config["epochs"],
        lr=config["learning_rate"], weight_decay=config["weight_decay"],
        diagnosis_head=_head_preset(config), checkpoint_metric=metric_from_config(config))


# Lấy nhãn model để in kết quả phù hợp với variant đang đánh giá.
def _model_label(config):
    return "M4-SG" if config["model"] == M4_SG_MODEL else "M4"


# Chặn output ghi đè checkpoint, predictions, manifest hoặc label mapping đầu vào.
def _protect_inputs(outputs, inputs):
    protected = {Path(path).resolve() for path in inputs}
    if any(Path(path).resolve() in protected for path in outputs):
        raise ValueError("Output must not overwrite its inputs")


# Kiểm tra protocol/head/loss metadata; trả các concept criteria để đánh giá.
def validate_m4_config(config):
    gradient_mode = _gradient_mode_from_config(config)
    metric_from_config(config)
    head_preset = _head_preset(config)
    expected = {**_variant_metadata(gradient_mode),
                "bottleneck": "categorical_onehot",
                "softmax_temperature": 1.0, "concept_weighting": "balanced",
                "backbone": "EfficientNet-B0", "diagnosis_head": DIAGNOSIS_HEADS[head_preset],
                "target_size": 224}
    if not isinstance(config, dict) or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Checkpoint is not the finalized M4 hard/ST protocol or M4 hard/SG protocol")
    if head_preset == "mlp128" and (config.get("diagnosis_hidden_dim") != 128 or
            config.get("diagnosis_dropout") != 0.3 or config.get("diagnosis_normalization") != "LayerNorm"):
        raise ValueError("Invalid M4 MLP128 diagnosis head specification")
    # Checkpoint cũ không có key này vẫn hợp lệ; runs mới phải lưu LR head rõ ràng.
    if "diagnosis_learning_rate" in config and (
            not np.isfinite(config["diagnosis_learning_rate"]) or config["diagnosis_learning_rate"] <= 0):
        raise ValueError("Invalid M4 diagnosis learning rate")
    weights = np.asarray(config.get("class_weights"), dtype=float)
    if weights.shape != (2,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Invalid M4 diagnosis class weights")
    if (any(type(config.get(key)) is not int for key in ("seed", "epochs", "batch_size")) or
            config["batch_size"] < 1 or config["epochs"] < 1 or
            config.get("augmentation_preset") not in {"legacy_letterbox", "comparison"} or
            not np.isfinite(config.get("concept_loss_weight", float("nan"))) or
            config["concept_loss_weight"] < 0 or
            not np.isfinite(config.get("dropout", float("nan"))) or not 0 <= config["dropout"] < 1):
        raise ValueError("Invalid M4 evaluation configuration")
    return shared.build_concept_criteria(config.get("concept_state_weights"), torch.device("cpu"))


# Eval/no_grad, chấm diagnosis/concepts và lưu chính hard vectors mà head đã nhận.
@torch.no_grad()
def evaluate(model, loader, diag_criterion, concept_criterion, concept_loss_weight,
             device, schema, threshold=0.5, train_observed_states=None):
    model.eval()
    diagnosis_logits, labels, vectors = [], [], []
    targets = {c: [] for c in CONCEPT_NAMES}
    probabilities = {c: [] for c in CONCEPT_NAMES}
    losses = np.zeros(3, dtype=float)
    for batch in loader:
        images, truth = batch["image"].to(device), batch["label"].to(device)
        indices = batch["concept_indices"].to(device)
        diag_logits, concept_logits, hard = model(images)
        diag_loss = diag_criterion(diag_logits, truth)
        concept_loss = shared._concept_loss(concept_logits, indices, concept_criterion)
        losses += len(truth) * np.array([(diag_loss + concept_loss_weight * concept_loss).item(),
                                        diag_loss.item(), concept_loss.item()])
        diagnosis_logits.append(diag_logits.cpu())
        labels.extend(truth.cpu().tolist())
        vectors.append(hard.cpu())
        for i, name in enumerate(CONCEPT_NAMES):
            targets[name].extend(indices[:, i].cpu().tolist())
            probabilities[name].append(concept_logits[name].softmax(-1).cpu().numpy())
    if not labels:
        raise ValueError("Cannot evaluate an empty split")
    hard = torch.cat(vectors).numpy()
    validate_hard_concepts(hard, schema)
    probabilities = {c: np.concatenate(probabilities[c]) for c in CONCEPT_NAMES}
    predictions = {}
    for name in CONCEPT_NAMES:
        start, k = schema["offsets"][name], schema["concept_num_classes"][name]
        predictions[name] = hard[:, start:start+k].argmax(-1)
        if not np.array_equal(predictions[name], probabilities[name].argmax(-1)):
            raise ValueError("Hard diagnosis inputs differ from concept probability argmax")
    probs = torch.cat(diagnosis_logits).softmax(-1)[:, 1].numpy()
    preds = (probs >= threshold).astype(int)
    concepts = {"true": {c: np.asarray(targets[c]) for c in CONCEPT_NAMES},
                "pred": predictions, "probabilities": probabilities, "hard": hard}
    return {"diagnosis_metrics": compute_metrics(labels, preds, probs),
            "concept_metrics": compute_concept_metrics(concepts["true"], predictions,
                CONCEPT_NUM_CLASSES, train_observed_states=train_observed_states),
            "loss": float(losses[0]/len(labels)),
            "diagnosis_loss": float(losses[1]/len(labels)),
            "concept_loss": float(losses[2]/len(labels)),
            "predictions": preds, "probabilities": probs, "concepts": concepts}


# Ghép predictions vào records từng ca và bổ sung hard_concept_vector để replay/intervention.
def _records(df, evaluation):
    rows = shared._extract_sample_predictions(df, evaluation["predictions"],
                                             evaluation["probabilities"], evaluation["concepts"])
    for row, vector in zip(rows, evaluation["concepts"]["hard"]):
        row["hard_concept_vector"] = vector.tolist()
    return rows


# Tạo metadata export: config, epoch, ngưỡng, paths, hashes và concept schema.
def _result(checkpoint, checkpoint_path, manifest_path):
    return {"model": checkpoint["config"]["model"], "hyperparameters": checkpoint["config"],
            "best_epoch": int(checkpoint["epoch"]),
            "decision_threshold": float(checkpoint["decision_threshold"]),
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": shared._manifest_hash(checkpoint_path),
            "manifest_sha256": checkpoint["manifest_sha256"],
            "manifest_fingerprint": manifest_fingerprint(manifest_path),
            "concept_schema": checkpoint["concept_schema"]}


# Chỉ train/validation: học -> best checkpoint -> ngưỡng validation -> JSON run.
def run_m4_experiment(manifest_path=None, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                      lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY,
                      concept_loss_weight=DEFAULT_CONCEPT_LOSS_WEIGHT, seed=DEFAULT_SEED,
                      device_name="auto", checkpoint_path=None, results_path=None,
                      save_results=True, augmentation_preset=DEFAULT_AUGMENTATION_PRESET,
                      overwrite=False, num_workers=2, gradient_mode="straight_through",
                      diagnosis_lr=None, diagnosis_head="linear",
                      checkpoint_metric=DEFAULT_CHECKPOINT_METRIC):
    if (any(type(value) is not int for value in (epochs, batch_size, num_workers, seed)) or
            epochs < 1 or batch_size < 1 or num_workers < 0 or not 0 <= seed < 2**32):
        raise ValueError("epochs/batch_size must be positive and num_workers non-negative")
    if not np.isfinite(lr) or lr <= 0 or not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if diagnosis_lr is not None and (not np.isfinite(diagnosis_lr) or diagnosis_lr <= 0):
        raise ValueError("diagnosis learning rate must be positive and finite")
    if diagnosis_head not in DIAGNOSIS_HEADS:
        raise ValueError("Unknown diagnosis head")
    if not np.isfinite(concept_loss_weight) or concept_loss_weight < 0:
        raise ValueError("concept_loss_weight must be finite and non-negative")
    if augmentation_preset not in {"legacy_letterbox", "comparison"}:
        raise ValueError("Unknown augmentation preset")
    metadata = _variant_metadata(gradient_mode)
    selection_name(checkpoint_metric)
    manifest_path = str(Path(manifest_path or ROOT / "data/manifest.csv").resolve())
    checkpoint_path, results_path = prepare_training_outputs(
        _default_paths(seed, augmentation_preset, concept_loss_weight, gradient_mode,
                       diagnosis_lr, epochs, lr, weight_decay, diagnosis_head, checkpoint_metric), checkpoint_path,
        results_path, overwrite, save_results)
    _protect_inputs([checkpoint_path, results_path],
                    [manifest_path, Path(manifest_path).parent / "label_mapping.json"])
    shared._prepare_output(checkpoint_path, overwrite)
    if save_results:
        shared._prepare_output(results_path, overwrite)
    shared.set_seed(seed)
    device = shared.get_device(device_name)
    schema = load_concept_schema(manifest_path)
    # Tập test chưa được tải trong bước học/chọn checkpoint: include_test=False.
    bundle = get_dataloaders(manifest_path=manifest_path, project_root=str(ROOT),
        batch_size=batch_size, num_workers=num_workers, target_size=224, augment_train=True,
        augmentation_preset=augmentation_preset, include_test=False)
    if len(bundle["datasets"]["train"]) != 413 or len(bundle["datasets"]["valid"]) != 203:
        raise ValueError("M4 requires the official 413/203 train/validation split")
    train_observed = shared.compute_train_observed_states(bundle["datasets"]["train"].df,
                                                         schema["label_mapping"])
    statistics = compute_concept_statistics(manifest_path, str(ROOT))
    state_weights = {c: statistics[c]["weights"] for c in CONCEPT_NAMES}
    class_weights = bundle["class_weights"].to(device)
    diag_criterion = nn.CrossEntropyLoss(weight=class_weights)
    concept_criterion = shared.build_concept_criteria(state_weights, device)
    model = _build_model(gradient_mode, CONCEPT_NAMES, CONCEPT_NUM_CLASSES,
                         pretrained=True, diagnosis_head=diagnosis_head).to(device)
    optimizer = _build_optimizer(model, lr, weight_decay, diagnosis_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    config = {**metadata, "bottleneck": "categorical_onehot", "softmax_temperature": 1.0,
        "backbone": "EfficientNet-B0", "pretrained_weights": "EfficientNet_B0_Weights.DEFAULT",
        "diagnosis_head": DIAGNOSIS_HEADS[diagnosis_head], "concept_heads": "Linear(1280, Ki) per concept",
        "epochs": epochs, "batch_size": batch_size, "learning_rate": lr,
        "weight_decay": weight_decay, "concept_loss_weight": concept_loss_weight, "seed": seed,
        "device": str(device), "diagnosis_loss": "Weighted_CrossEntropy",
        "concept_loss": "Multi-Head_CrossEntropy_balanced_mean", "concept_weighting": "balanced",
        "concept_state_weights": state_weights, "concept_train_statistics": statistics,
        "dropout": 0.2, "class_weights": class_weights.cpu().tolist(),
        "augmentation_preset": augmentation_preset, "target_size": 224, "num_workers": num_workers,
        "scheduler": "CosineAnnealingLR(eta_min=1e-6)",
        "checkpoint_selection": selection_name(checkpoint_metric),
        "threshold_selection": "maximize_validation_balanced_accuracy_tie_nearest_0.5",
        "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
        "numpy_version": str(np.__version__), "pandas_version": str(pd.__version__),
        "scikit_learn_version": str(sklearn.__version__), "platform": platform.platform()}
    if diagnosis_lr is not None:
        config["diagnosis_learning_rate"] = diagnosis_lr
        config["optimizer_parameter_groups"] = "concept_predictor_and_diagnosis_head"
    if diagnosis_head == "mlp128":
        config.update(diagnosis_hidden_dim=128, diagnosis_dropout=0.3,
                      diagnosis_normalization="LayerNorm")
    validate_m4_config(config)
    fingerprint, raw_hash = manifest_fingerprint(manifest_path), shared._manifest_hash(manifest_path)
    best_score, history = -1., []
    print(f"{_model_label(config)} Hard CBM | gradient={gradient_mode} | seed={seed} | device={device} | epochs={epochs}")
    for epoch in range(1, epochs+1):
        # Ghi LR thực sự dùng ở epoch này, trước khi scheduler chuyển sang epoch kế.
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        train_loss, train_diag, train_concept, train_bacc = shared.train_one_epoch(
            model, bundle["train"], diag_criterion, concept_criterion, concept_loss_weight, optimizer, device)
        val = evaluate(model, bundle["valid"], diag_criterion, concept_criterion,
                       concept_loss_weight, device, schema, train_observed_states=train_observed)
        scheduler.step()
        # Chọn epoch bằng checkpoint_metric @0.5: default Macro F1, study cũ BAcc.
        # History luôn ghi cả hai metrics, nên có thể đối chiếu tiêu chí đã dùng.
        # Khi hòa, phép so sánh > giữ epoch tốt nhất xuất hiện đầu tiên.
        score = selection_score(val["diagnosis_metrics"], checkpoint_metric)
        history.append({"epoch": epoch, "train_loss": train_loss,
            "train_diagnosis_loss": train_diag, "train_concept_loss": train_concept,
            "train_diagnosis_balanced_accuracy": train_bacc, "validation_loss": val["loss"],
            "validation_diagnosis_loss": val["diagnosis_loss"], "validation_concept_loss": val["concept_loss"],
            "validation_balanced_accuracy_at_0.5": val["diagnosis_metrics"]["balanced_accuracy"],
            "validation_f1_macro_at_0.5": val["diagnosis_metrics"]["f1_macro"],
            "learning_rates": learning_rates,
            "validation_f1_melanoma_at_0.5": val["diagnosis_metrics"]["f1_melanoma"],
            "validation_roc_auc": val["diagnosis_metrics"]["roc_auc"],
            "validation_pr_auc": val["diagnosis_metrics"]["pr_auc"],
            "validation_concept_f1_macro_all_defined": val["concept_metrics"]["overall_f1_macro_all_defined"]})
        if score > best_score:
            best_score = score
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "config": config,
                "val_balanced_acc": val["diagnosis_metrics"]["balanced_accuracy"],
                "val_f1_macro": val["diagnosis_metrics"]["f1_macro"],
                "selection_score": score, "val_metrics": val["diagnosis_metrics"],
                "val_concept_metrics": val["concept_metrics"], "manifest_sha256": raw_hash,
                "manifest_fingerprint": fingerprint, "concept_schema": schema}, checkpoint_path)
        print(f"Epoch {epoch:02d}/{epochs} | Train loss={train_loss:.4f} "
              f"(Diag={train_diag:.4f}, Concept={train_concept:.4f}, BAcc={100*train_bacc:.2f}%) | "
              f"Valid loss={val['loss']:.4f} | Valid {checkpoint_metric}@0.5={score:.4f} | "
              f"Valid AUC={val['diagnosis_metrics']['roc_auc']:.4f} | "
              f"Valid Concept F1={val['concept_metrics']['overall_f1_macro_all_defined']:.4f}",
              flush=True)
    # Train xong nạp best epoch, không dùng mặc nhiên weights của epoch cuối.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    val = evaluate(model, bundle["valid"], diag_criterion, concept_criterion,
                   concept_loss_weight, device, schema, train_observed_states=train_observed)
    # Chọn ngưỡng trên validation của best epoch: tối đa BAcc, hòa gần 0.5.
    # Lưu ngưỡng vào checkpoint để test/intervention dùng lại, không tune trên test.
    threshold, _ = shared.select_validation_threshold(
        bundle["datasets"]["valid"].df.diagnosis_binary.to_numpy(dtype=int), val["probabilities"])
    val_preds = (val["probabilities"] >= threshold).astype(int)
    val_metrics = compute_metrics(bundle["datasets"]["valid"].df.diagnosis_binary, val_preds, val["probabilities"])
    checkpoint.update(decision_threshold=threshold, validation_metrics_at_threshold=val_metrics)
    torch.save(checkpoint, checkpoint_path)
    summary = print_metrics_table(val_metrics, title=f"{_model_label(config)} validation (epoch={checkpoint['epoch']}, threshold={threshold:.6f})")
    summary += "\n\n" + print_concept_metrics_table(val["concept_metrics"])
    result = _result(checkpoint, checkpoint_path, manifest_path)
    result.update(mode="train_validation",
                  validation_metrics=val_metrics, validation_concept_metrics=val["concept_metrics"],
                  validation_loss=val["loss"], train_observed_states=train_observed, history=history, summary=summary)
    result["validation_metrics_at_0.5"] = val["diagnosis_metrics"]
    val["predictions"] = val_preds
    result["validation_predictions"] = _records(bundle["datasets"]["valid"].df, val)
    if save_results:
        save_experiment_results(results_path, validation=result, overwrite=overwrite)
    return result


# Nạp frozen model/ngưỡng, kiểm tra official test/schema rồi chấm 395 ca test.
def evaluate_m4_test(checkpoint_path, manifest_path=None, device_name="auto",
                     results_path=None, save_results=True, overwrite=False, expected_gradient_mode=None):
    manifest_path = str(Path(manifest_path or ROOT / "data/manifest.csv").resolve())
    device = shared.get_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint.get("config")
    validate_m4_config(config)
    _require_gradient_mode(config, expected_gradient_mode)
    gradient_mode = _gradient_mode_from_config(config)
    validate_manifest(checkpoint, manifest_path)
    schema = validate_concept_schema(checkpoint, manifest_path)
    threshold = float(checkpoint.get("decision_threshold", float("nan")))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("M4 checkpoint has no valid frozen validation threshold")
    results_path = results_path or _config_paths(config, gradient_mode)[1]
    if save_results:
        _protect_inputs([results_path], [checkpoint_path, manifest_path, Path(manifest_path).parent / "label_mapping.json"])
        shared._prepare_output(results_path, overwrite)
    df = pd.read_csv(manifest_path)
    observed = shared.compute_train_observed_states(df[df.split == "train"], schema["label_mapping"])
    dataset = Derm7ptDataset(manifest_path=manifest_path, project_root=str(ROOT), split="test",
        label_mapping=schema["label_mapping"], transform=get_transforms("valid", target_size=config["target_size"], augment=False))
    if len(dataset) != 395:
        raise ValueError("M4 requires the official 395-case test split")
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    model = _build_model(gradient_mode, schema["concept_names"], schema["concept_num_classes"],
                         pretrained=False, dropout=config["dropout"],
                         diagnosis_head=_head_preset(config)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    val = evaluate(model, loader, nn.CrossEntropyLoss(weight=torch.tensor(config["class_weights"], device=device)),
                   shared.build_concept_criteria(config["concept_state_weights"], device),
                   config["concept_loss_weight"], device, schema, threshold, observed)
    result = _result(checkpoint, checkpoint_path, manifest_path)
    result.update(mode="final_test", test_metrics=val["diagnosis_metrics"],
        test_concept_metrics=val["concept_metrics"], test_loss=val["loss"],
        test_predictions=_records(dataset.df, val),
        summary=print_metrics_table(val["diagnosis_metrics"], title=f"{_model_label(config)} test (threshold={threshold:.6f})"))
    result["summary"] += "\n\n" + print_concept_metrics_table(val["concept_metrics"])
    if save_results:
        save_experiment_results(results_path, test=result, overwrite=overwrite)
    return result


# Đối chiếu cases/labels/GT/hard vectors với manifest và tính lại concept metrics.
def validate_m4_test_export(export, manifest_path):
    validate_m4_config(export.get("hyperparameters"))
    validate_manifest(export, manifest_path)
    schema = validate_concept_schema(export, manifest_path)
    if export.get("model") != export["hyperparameters"]["model"] or export.get("mode") not in {"final_test", "train_validation_test"}:
        raise ValueError("Expected frozen M4 test predictions")
    test = pd.read_csv(manifest_path)
    test = test[test.split == "test"].set_index("case_num")
    rows = export.get("test_predictions", [])
    ids = [row["case_num"] for row in rows]
    if len(ids) != 395 or len(ids) != len(set(ids)) or set(ids) != set(test.index):
        raise ValueError("Prediction case IDs differ from the official test split")
    vectors, true_indices, labels = [], [], []
    for row in rows:
        case = test.loc[row["case_num"]]
        if row["y_true"] != int(case.diagnosis_binary):
            raise ValueError("Prediction labels differ from manifest")
        targets, hard_parts = [], []
        if any(k not in row for k in ("concept_true", "concept_pred", "concept_probabilities", "hard_concept_vector")):
            raise ValueError("Export has no hard per-case concepts")
        for name in schema["concept_names"]:
            k = schema["concept_num_classes"][name]
            target = schema["label_mapping"][name][case[name]]
            if row["concept_true"][name] != target:
                raise ValueError("Prediction concept targets differ from manifest")
            probs = np.asarray(row["concept_probabilities"][name], dtype=float)
            if (probs.shape != (k,) or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)) or
                    not np.isclose(probs.sum(), 1., atol=1e-5, rtol=0)):
                raise ValueError("Invalid export concept probability block")
            index = int(probs.argmax())
            if row["concept_pred"][name] != index:
                raise ValueError("Export concept indices differ from probability argmax")
            hard_parts.extend(np.eye(k, dtype=int)[index].tolist())
            targets.append(target)
        if not np.array_equal(np.asarray(row["hard_concept_vector"]), hard_parts):
            raise ValueError("Export hard vectors differ from probability argmax")
        vectors.append(row["hard_concept_vector"])
        true_indices.append(targets)
        labels.append(row["y_true"])

    frame = pd.read_csv(manifest_path)
    observed = shared.compute_train_observed_states(frame[frame.split == "train"], schema["label_mapping"])
    verified = compute_concept_metrics(
        {c: np.asarray(true_indices)[:, i] for i, c in enumerate(schema["concept_names"])},
        {c: np.asarray([row["concept_pred"][c] for row in rows]) for c in schema["concept_names"]},
        schema["concept_num_classes"], train_observed_states=observed)
    if not _metrics_match(export.get("test_concept_metrics"), verified):
        raise ValueError("Export concept metrics do not match per-case predictions")
    return vectors, true_indices, labels


# So metrics dict/list lồng nhau với sai số số thực cho phép, dùng kiểm tra export.
def _metrics_match(saved, verified):
    if isinstance(verified, dict):
        return (isinstance(saved, dict) and set(saved) == set(verified) and
                all(_metrics_match(saved[key], value) for key, value in verified.items()))
    if isinstance(verified, list):
        return (isinstance(saved, list) and len(saved) == len(verified) and
                all(_metrics_match(a, b) for a, b in zip(saved, verified)))
    try:
        return bool(np.isclose(saved, verified, atol=1e-10, rtol=1e-7))
    except (TypeError, ValueError):
        return False


# Kiểm tra checkpoint/export, dựng frozen head, sửa GT concept groups và xuất JSON riêng.
def run_m4_intervention(checkpoint_path, predictions_path, manifest_path=None,
                        results_path=None, overwrite=False, expected_seed=None, expected_gradient_mode=None):
    manifest_path = str(Path(manifest_path or ROOT / "data/manifest.csv").resolve())
    results_path = results_path or str(Path(predictions_path).parent / "intervention" / Path(predictions_path).name)
    _protect_inputs([results_path], [checkpoint_path, predictions_path, manifest_path,
                                   Path(manifest_path).parent / "label_mapping.json"])
    shared._prepare_output(results_path, overwrite)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_m4_config(checkpoint.get("config"))
    if expected_gradient_mode is not None:
        _variant_metadata(expected_gradient_mode)
        if _gradient_mode_from_config(checkpoint["config"]) != expected_gradient_mode:
            raise ValueError("Requested gradient mode differs from paired checkpoint/export")
    if expected_seed is not None and checkpoint["config"]["seed"] != expected_seed:
        raise ValueError("Requested seed differs from paired checkpoint/export")
    validate_manifest(checkpoint, manifest_path)
    schema = validate_concept_schema(checkpoint, manifest_path)
    export = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    if export.get("model") != checkpoint["config"]["model"] or export.get("mode") not in {"final_test", "train_validation_test"}:
        raise ValueError("Expected frozen M4 test predictions")
    validate_manifest(export, manifest_path)
    validate_concept_schema(export, manifest_path)
    if export.get("hyperparameters") != checkpoint["config"]:
        raise ValueError("Prediction training config differs from checkpoint")
    checkpoint_hash = shared._manifest_hash(checkpoint_path)
    if export.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Prediction export does not belong to this checkpoint")
    threshold = float(checkpoint.get("decision_threshold", float("nan")))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1 or export.get("decision_threshold") != threshold:
        raise ValueError("Prediction threshold differs from frozen checkpoint")
    vectors, true_indices, labels = validate_m4_test_export(export, manifest_path)
    rows = export["test_predictions"]
    ids = [row["case_num"] for row in rows]
    # Intervention dựng đúng head đã train; MLP cần cả Linear, LayerNorm và biases.
    head = make_diagnosis_head(schema["total_states"], 2, _head_preset(checkpoint["config"]))
    state = checkpoint["model_state_dict"]
    head.load_state_dict({name.removeprefix("diagnosis_head."): value
                          for name, value in state.items() if name.startswith("diagnosis_head.")})
    # Evaluator sửa bản sao concept vectors; không sửa annotations/dữ liệu nguồn.
    # Threshold giữ cố định; subset rỗng phải tái hiện diagnosis baseline đã lưu.
    result = evaluate_hard_interventions(head, vectors, true_indices, labels, schema, threshold, ids,
                                        baseline_probabilities=[row["y_prob"] for row in rows])
    if not np.allclose(result["baseline_y_prob"], [r["y_prob"] for r in rows], atol=1e-6, rtol=1e-5):
        raise ValueError("Export hard vectors do not reproduce baseline diagnosis")
    if result["baseline_y_pred"] != [r["y_pred"] for r in rows]:
        raise ValueError("Export diagnosis predictions do not use the frozen threshold")
    result.update(model=checkpoint["config"]["model"], mode="intervention_analysis", hyperparameters=checkpoint["config"],
        input_representation="categorical_onehot", concept_schema=schema,
        manifest_fingerprint=manifest_fingerprint(manifest_path), checkpoint_sha256=checkpoint_hash,
        checkpoint_epoch=int(checkpoint["epoch"]), checkpoint_path=str(Path(checkpoint_path).resolve()),
        predictions_path=str(Path(predictions_path).resolve()),
        evaluation_environment={"device": "cpu", "platform": platform.platform(),
            "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
            "scikit_learn_version": str(sklearn.__version__)})
    path = Path(results_path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {_model_label(checkpoint['config'])} intervention: {len(rows)} cases × {result['num_subsets']} subsets → {path}")
    return result


# Điều phối mode train/test/intervention cho cả ST/SG; skip_test chỉ hợp lệ khi train.
def run_cli(gradient_mode, argv=None):
    _variant_metadata(gradient_mode)
    label = "M4-SG" if gradient_mode == "stop_gradient" else "M4-ST"
    parser = argparse.ArgumentParser(description=f"{label}: train → validation → frozen test, or --skip_test pilot.")
    add_selection_argument(parser)
    parser.add_argument("--mode", choices=["train", "test", "intervention"], default="train")
    parser.add_argument("--skip_test", action="store_true", help="Pilot: train/validation only; valid with --mode train")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--diagnosis_lr", type=float,
                        help="LR riêng cho diagnosis head; không truyền thì dùng --lr cho toàn model")
    parser.add_argument("--diagnosis_head", choices=list(DIAGNOSIS_HEADS), default="linear",
                        help="linear: baseline; mlp128: Linear/LayerNorm/ReLU/dropout/Linear")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--concept_loss_weight", type=float, default=DEFAULT_CONCEPT_LOSS_WEIGHT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--augmentation_preset", choices=["legacy_letterbox", "comparison"], default=DEFAULT_AUGMENTATION_PRESET)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--manifest_path")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--results_path")
    parser.add_argument("--predictions_path")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args(argv)
    if args.skip_test and args.mode != "train":
        parser.error("--skip_test is only valid with --mode train")
    defaults = _default_paths(args.seed, args.augmentation_preset, args.concept_loss_weight, gradient_mode,
                             args.diagnosis_lr, args.epochs, args.lr, args.weight_decay, args.diagnosis_head,
                             args.checkpoint_metric)
    checkpoint_path, results_path = args.checkpoint_path or defaults[0], args.results_path or defaults[1]
    if args.mode == "train":
        validation = run_m4_experiment(manifest_path=args.manifest_path, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
            concept_loss_weight=args.concept_loss_weight, seed=args.seed, device_name=args.device,
            checkpoint_path=checkpoint_path, results_path=results_path, save_results=not args.no_save,
            augmentation_preset=args.augmentation_preset, overwrite=args.overwrite, num_workers=args.num_workers,
            gradient_mode=gradient_mode, diagnosis_lr=args.diagnosis_lr, diagnosis_head=args.diagnosis_head,
            checkpoint_metric=args.checkpoint_metric)
        if args.skip_test:
            return validation
        test = evaluate_m4_test(checkpoint_path, args.manifest_path, args.device, save_results=False,
                                expected_gradient_mode=gradient_mode)
        if not args.no_save:
            return save_experiment_results(results_path, validation=validation, test=test, overwrite=True)
        return {**validation, **test, "mode": "train_validation_test"}
    if args.mode == "test":
        return evaluate_m4_test(checkpoint_path, args.manifest_path, args.device,
                                args.results_path, not args.no_save, args.overwrite,
                                expected_gradient_mode=gradient_mode)
    if args.no_save:
        parser.error("--no_save is not supported for intervention mode")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_m4_config(checkpoint.get("config"))
    config = checkpoint["config"]
    predictions_path = args.predictions_path or _config_paths(config, _gradient_mode_from_config(config))[1]
    return run_m4_intervention(checkpoint_path, predictions_path, args.manifest_path,
                               args.results_path, args.overwrite, expected_gradient_mode=gradient_mode)
