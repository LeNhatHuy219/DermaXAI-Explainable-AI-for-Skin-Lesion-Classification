# Chạy intervention M3 trên nhiều seeds, rồi tổng hợp bảng và biểu đồ.
# Đầu vào: frozen M3 checkpoints, test exports và GT concepts đã đối chiếu manifest.
# Thử đủ 128 subsets của bảy groups; sửa cả group bằng GT one-hot, giữ ngưỡng cố định.
# Tính metric từng subset -> mean theo số groups sửa -> mean/sample SD giữa seeds.
# Đầu ra: results/<metric>/m3/<cấu hình>/intervention/, hoặc --output_dir.
# Gồm JSON từng seed/summary, curve.csv, per_concept.csv, full_intervention_cases.csv,
# intervention_curve.png/.svg và report.md; không train hay sửa annotation nguồn.
# summarize_interventions()/export_report() còn được M4 tái sử dụng để xuất báo cáo.

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Preserve the project's torch-before-sklearn import order.
from experiments.run_m3 import (DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_WEIGHT_DECAY,
                                DIAGNOSIS_HEADS, _default_paths, run_m3_intervention)
from src.metrics import compute_metrics
from src.selection import add_selection_argument
import numpy as np

METRICS = ("accuracy", "balanced_accuracy", "f1_macro", "sensitivity",
           "specificity", "roc_auc", "pr_auc")
IGNORE_CONFIG = {"seed", "device", "num_workers", "platform", "torch_version",
                 "torchvision_version", "numpy_version", "pandas_version",
                 "scikit_learn_version"}


# Trả values, mean và sample SD giữa seeds; đây không phải CI theo ca test.
def stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
            "values": values.tolist()}


# Kiểm tra exports, chấm từng subset, lấy mean theo m groups rồi tổng hợp giữa seeds.
def summarize_interventions(runs):
    if len(runs) < 3:
        raise ValueError("At least three distinct seeds are required")
    first = runs[0]
    names = first["concept_schema"]["concept_names"]
    count = len(names)
    config = {k: v for k, v in first["hyperparameters"].items() if k not in IGNORE_CONFIG}
    seeds, labels = set(), dict(zip(first["case_ids"], first["y_true"]))
    curves, singles, areas = [], [], []
    for run in runs:
        seed = run["hyperparameters"]["seed"]
        if seed in seeds:
            raise ValueError("Duplicate seed")
        seeds.add(seed)
        if run.get("mode") != "intervention_analysis" or run.get("model") != first["model"]:
            raise ValueError("Expected intervention exports of the same model")
        if {k: v for k, v in run["hyperparameters"].items() if k not in IGNORE_CONFIG} != config:
            raise ValueError("Training protocols differ")
        if run["concept_schema"]["sha256"] != first["concept_schema"]["sha256"]:
            raise ValueError("Concept schemas differ")
        if run["manifest_fingerprint"]["normalized_sha256"] != first["manifest_fingerprint"]["normalized_sha256"]:
            raise ValueError("Manifests differ")
        ids = run["case_ids"]
        truth = np.asarray(run["y_true"], dtype=int)
        if len(set(ids)) != len(ids) or dict(zip(ids, truth.tolist())) != labels or len(truth) != len(ids):
            raise ValueError("Test cases or labels differ")
        threshold = run["decision_threshold"]
        baseline = np.asarray(run["baseline_y_pred"], dtype=int)
        base_probs = np.asarray(run["baseline_y_prob"], dtype=float)
        if (not np.isfinite(threshold) or not 0 <= threshold <= 1 or
                baseline.shape != truth.shape or base_probs.shape != truth.shape or
                not np.isfinite(base_probs).all() or np.any((base_probs < 0) | (base_probs > 1)) or
                not np.array_equal(baseline, (base_probs >= threshold).astype(int))):
            raise ValueError("Invalid frozen baseline")
        base_metrics = compute_metrics(truth, baseline, base_probs)
        subsets, seen = [], set()
        for row in run["subsets"]:
            groups = frozenset(row["groups"])
            if (len(groups) != len(row["groups"]) or not groups <= set(names) or groups in seen or
                    row["m"] != len(groups) or not np.isclose(row["rate"], len(groups)/count)):
                raise ValueError("Invalid or duplicate intervention subset")
            seen.add(groups)
            probs = np.asarray(row["y_prob"], dtype=float)
            preds = np.asarray(row["y_pred"], dtype=int)
            if (probs.shape != truth.shape or preds.shape != truth.shape or
                    not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)) or
                    not np.array_equal(preds, (probs >= threshold).astype(int))):
                raise ValueError("Subset predictions do not use the frozen threshold")
            verified = compute_metrics(truth, preds, probs)
            if any(not np.isclose(verified[k], row["diagnosis_metrics"][k], atol=1e-10) for k in METRICS):
                raise ValueError("Subset metrics differ from predictions")
            if not groups and (not np.array_equal(preds, baseline) or not np.allclose(probs, base_probs)):
                raise ValueError("m=0 differs from baseline")
            improved = [ids[i] for i in np.flatnonzero((baseline != truth) & (preds == truth))]
            worsened = [ids[i] for i in np.flatnonzero((baseline == truth) & (preds != truth))]
            if improved != row["improved_case_ids"] or worsened != row["worsened_case_ids"]:
                raise ValueError("Case transitions differ from predictions")
            if not np.allclose(row["probability_delta"], probs-base_probs, atol=1e-7):
                raise ValueError("Probability deltas differ from baseline")
            remaining = row["remaining_concept_accuracy"]
            if (len(groups) == count and remaining is not None) or (len(groups) < count and
                    (remaining is None or not np.isfinite(remaining) or not 0 <= remaining <= 1)):
                raise ValueError("Invalid remaining-concept accuracy")
            subsets.append({"m": len(groups), "groups": row["groups"], "metrics": verified,
                            "improved_cases": len(improved), "worsened_cases": len(worsened),
                            "net_correct_cases": len(improved)-len(worsened),
                            "unchanged_diagnosis_cases": int(np.sum(preds == baseline)),
                            "remaining_concept_accuracy": remaining})
        if len(seen) != 2**count or run["num_subsets"] != len(seen):
            raise ValueError("Expected all intervention subsets")
        curve = []
        for m in range(count+1):
            selected = [row for row in subsets if row["m"] == m]
            if len(selected) != math.comb(count, m):
                raise ValueError("Incomplete intervention level")
            entry = {"m": m, "rate": m/count, "num_subsets": len(selected)}
            for key in METRICS:
                values = [row["metrics"][key] for row in selected]
                entry[key] = float(np.mean(values))
                entry["delta_"+key] = entry[key]-base_metrics[key]
                entry[key+"_subset_sd"] = float(np.std(values, ddof=0))
            for key in ("improved_cases", "worsened_cases", "net_correct_cases", "unchanged_diagnosis_cases"):
                entry[key] = float(np.mean([row[key] for row in selected]))
            entry["remaining_concept_accuracy"] = (float(np.mean([row["remaining_concept_accuracy"]
                                                                 for row in selected])) if m < count else None)
            curve.append(entry)
        curves.append(curve)
        singles.append({row["groups"][0]: row for row in subsets if row["m"] == 1})
        integrate = getattr(np, "trapezoid", None)
        if integrate is None:
            integrate = np.trapz
        area = {}
        for key in ("balanced_accuracy", "f1_macro"):
            absolute = float(integrate([row[key] for row in curve], x=[row["rate"] for row in curve]))
            area[key] = {"absolute": absolute, "gain_over_baseline": absolute-base_metrics[key]}
            stored = run["intervention_curve_auc"][key]
            if any(not np.isclose(stored[k], area[key][k], atol=1e-10) for k in area[key]):
                raise ValueError("Intervention AUC differs from predictions")
        areas.append(area)
    numeric = list(METRICS) + ["delta_"+k for k in METRICS] + [
        "improved_cases", "worsened_cases", "net_correct_cases", "unchanged_diagnosis_cases"]
    combined = []
    for m in range(count+1):
        row = {"m": m, "rate": m/count, "num_subsets": math.comb(count, m)}
        row.update({key: stats([curve[m][key] for curve in curves]) for key in numeric})
        row["subset_population_sd_by_seed"] = {
            key: [curve[m][key+"_subset_sd"] for curve in curves] for key in METRICS}
        row["remaining_concept_accuracy"] = (stats([curve[m]["remaining_concept_accuracy"]
                                                    for curve in curves]) if m < count else None)
        combined.append(row)
    per_concept = {}
    for name in names:
        per_concept[name] = {}
        for key in ("balanced_accuracy", "f1_macro"):
            per_concept[name]["delta_"+key] = stats([
                singles[i][name]["metrics"][key]-curves[i][0][key] for i in range(len(runs))])
        for key in ("improved_cases", "worsened_cases", "net_correct_cases"):
            per_concept[name][key] = stats([single[name][key] for single in singles])
    return {"model": first["model"], "protocol": "exhaustive_ground_truth_group_intervention_v1",
            "seeds": [run["hyperparameters"]["seed"] for run in runs], "num_cases": len(labels),
            "num_subsets_per_seed": 2**count, "training_protocol": config,
            "concept_schema": first["concept_schema"], "manifest_fingerprint": first["manifest_fingerprint"],
            "decision_thresholds": [run["decision_threshold"] for run in runs],
            "evaluation_environments": [run.get("evaluation_environment", {}) for run in runs],
            "checkpoint_sha256_by_seed": {str(run["hyperparameters"]["seed"]): run["checkpoint_sha256"] for run in runs},
            "curve": combined, "per_concept": per_concept,
            "intervention_curve_auc": {key: {kind: stats([a[key][kind] for a in areas])
                                            for kind in ("absolute", "gain_over_baseline")}
                                       for key in ("balanced_accuracy", "f1_macro")},
            "averaging": "metrics per subset on all test cases, mean across subsets at m, then mean and sample SD across seeds",
            "uncertainty": "Seed sample SD and subset population SD are separate; neither is a paired bootstrap confidence interval."}


# Ghi list records thành CSV có header, dùng cho các bảng/ca intervention.
def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Xuất summary/CSV/biểu đồ/report và nhãn model đúng; hàm dùng chung với M4.
def export_report(summary, runs, output_dir, model_label="M3"):
    expected = {"M3": "M3_SoftJointCBM", "M4": "M4_HardJointCBM", "M4-ST": "M4_HardJointCBM",
                "M4-SG": "M4_HardStopGradientCBM"}
    if model_label not in expected or summary["model"] != expected[model_label]:
        raise ValueError("Report model label differs from intervention summary")
    output_dir = Path(output_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for entry in summary["curve"]:
        row = {k: entry[k] for k in ("m", "rate", "num_subsets")}
        for key, value in entry.items():
            if isinstance(value, dict) and "mean" in value:
                row[key+"_mean"] = value["mean"]
                row[key+"_seed_sd"] = value["sample_sd"]
        rows.append(row)
    write_csv(output_dir / "curve.csv", rows)
    concept_rows = []
    for name, entry in summary["per_concept"].items():
        row = {"concept": name}
        for key, value in entry.items():
            row[key+"_mean"] = value["mean"]
            row[key+"_seed_sd"] = value["sample_sd"]
        concept_rows.append(row)
    write_csv(output_dir / "per_concept.csv", concept_rows)
    cases = []
    for run in runs:
        source = json.loads(Path(run["predictions_path"]).read_text(encoding="utf-8"))
        metadata = {row["case_num"]: row for row in source["test_predictions"]}
        full = next(row for row in run["subsets"] if row["m"] == 7)
        for i, case_id in enumerate(run["case_ids"]):
            before, after, truth = run["baseline_y_pred"][i], full["y_pred"][i], run["y_true"][i]
            status = "improved" if before != truth and after == truth else (
                "worsened" if before == truth and after != truth else "unchanged")
            cases.append({"seed": run["hyperparameters"]["seed"], "case_num": case_id,
                          "y_true": truth, "baseline_y_pred": before, "corrected_y_pred": after,
                          "baseline_y_prob": run["baseline_y_prob"][i], "corrected_y_prob": full["y_prob"][i],
                          "probability_delta": full["probability_delta"][i], "outcome": status,
                          "is_inconsistent_profile": metadata[case_id]["is_inconsistent"]})
    write_csv(output_dir / "full_intervention_cases.csv", cases)

    # Use a writable task-local cache and a headless backend.
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    xs = [row["m"] for row in summary["curve"]]
    for ax, key, label in zip(axes, ("balanced_accuracy", "f1_macro"), ("Balanced accuracy", "Macro F1")):
        means = np.array([row[key]["mean"] for row in summary["curve"]])
        sd = np.array([row[key]["sample_sd"] for row in summary["curve"]])
        for run in runs:
            ax.plot(xs, [row["diagnosis_metrics"][key]["mean"] for row in run["curve"]],
                    linewidth=1, alpha=.5, label=f"Seed {run['hyperparameters']['seed']}")
        ax.fill_between(xs, means-sd, means+sd, color="#2563eb", alpha=.15, label="Mean ± seed SD")
        ax.plot(xs, means, color="#2563eb", marker="o", linewidth=2.5, label="Mean across seeds")
        ax.axhline(means[0], color="#64748b", linestyle="--", linewidth=1, label="Baseline mean")
        ax.set(xlabel="Number of corrected concept groups", ylabel=label, xticks=xs, ylim=(0, 1))
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{model_label} ground-truth intervention · {summary['num_cases']} test cases · {len(runs)} seeds\n"
                 "Mean over all subsets at each m; shading shows sample SD across seeds", fontsize=11)
    fig.savefig(output_dir / "intervention_curve.png", dpi=200)
    fig.savefig(output_dir / "intervention_curve.svg")
    plt.close(fig)
    lines = [f"# Kết quả concept intervention của {model_label}", "",
             f"Seeds: {summary['seeds']}; {summary['num_cases']} ca test; 128 subsets cho mỗi seed.", "",
             "Thay toàn bộ group được sửa bằng ground-truth one-hot. Giữ nguyên các groups chưa sửa, "
             "diagnosis head và threshold validation của từng seed; không huấn luyện lại.", "",
             "Metrics được tính riêng cho mỗi subset trên toàn bộ test, rồi lấy trung bình theo m, "
             "sau đó tổng hợp mean ± sample SD giữa seeds. SD này không phải khoảng tin cậy theo mẫu.", "",
             "| Số groups sửa | Balanced Accuracy (%) | Macro-F1 | Ca tốt lên (TB) | Ca xấu đi (TB) |",
             "|---|---:|---:|---:|---:|"]
    for row in summary["curve"]:
        ba, f1 = row["balanced_accuracy"], row["f1_macro"]
        lines.append(f"| {row['m']} | {100*ba['mean']:.2f} ± {100*ba['sample_sd']:.2f} | "
                     f"{f1['mean']:.4f} ± {f1['sample_sd']:.4f} | {row['improved_cases']['mean']:.2f} | "
                     f"{row['worsened_cases']['mean']:.2f} |")
    lines.extend(["", "![Intervention curve](intervention_curve.png)", "",
                  "Ca tốt lên/xấu đi được so với m=0 của cùng seed; ở m=1…6 là trung bình qua subsets và seeds, "
                  "không phải số ca duy nhất. Ở m=7, mỗi seed có một subset sửa đủ bảy groups.", "",
                  "| Sửa toàn bộ 7 groups | Balanced Accuracy (%) | Macro-F1 | Tốt lên | Xấu đi |",
                  "|---|---:|---:|---:|---:|"])
    for run in runs:
        full = next(row for row in run["subsets"] if row["m"] == 7)
        metrics = full["diagnosis_metrics"]
        lines.append(f"| Seed {run['hyperparameters']['seed']} | {100*metrics['balanced_accuracy']:.2f} | "
                     f"{metrics['f1_macro']:.4f} | {len(full['improved_case_ids'])} | {len(full['worsened_case_ids'])} |")
    baseline, corrected = summary["curve"][0], summary["curve"][-1]
    delta = 100*corrected["delta_balanced_accuracy"]["mean"]
    interpretation = (
        "Diagnosis head M3 được học từ soft probabilities; khi intervention, input của head "
        "chuyển sang ground-truth one-hot ở các groups được sửa. Sự khác biệt representation là một giả thuyết "
        "cần khảo sát để giải thích kết quả, chưa được xác lập là nguyên nhân. Các số liệu này chưa tự chứng minh "
        "concept leakage, ảnh hưởng của inconsistency hoặc ưu thế của ECBM."
        if model_label == "M3" else
        f"Diagnosis head {model_label} được học từ predicted one-hot concepts và nhận ground-truth one-hot khi intervention. "
        "Hai đầu vào cùng dạng rời rạc, nhưng phân bố tổ hợp concept dự đoán và concept GT vẫn có thể khác nhau. "
        "Cùng representation không bảo đảm hiệu chỉnh có lợi. Kết quả chưa tự xác lập nguyên nhân "
        "của chênh lệch M3–M4, ảnh hưởng của inconsistency hoặc ưu thế của ECBM.")
    if model_label == "M4-SG":
        interpretation += (" Khi huấn luyện M4-SG, gradient của loss chẩn đoán dừng tại bottleneck; "
                           "concept heads và backbone được cập nhật bằng concept loss.")
    lines.extend(["", f"Sửa đủ bảy groups làm Balanced Accuracy trung bình thay đổi {delta:+.2f} điểm phần trăm "
                  f"({100*baseline['balanced_accuracy']['mean']:.2f}% → "
                  f"{100*corrected['balanced_accuracy']['mean']:.2f}%). Đây là kết quả mô tả trên các checkpoints hiện có.",
                  "", interpretation])
    lines.extend(["", "Các file đi kèm: `summary.json`, `curve.csv`, `per_concept.csv`, "
                  "`full_intervention_cases.csv` và JSON chi tiết cho từng seed.", "",
                  "`per_concept.csv` đo tác động sửa riêng từng group, không đo tầm quan trọng tổng quát hoặc quan hệ nhân quả. "
                  f"Với {model_label}, groups chưa sửa giữ nguyên dự đoán nên accuracy của chúng không thay đổi khi giữ cùng group. "
                  "Trung bình qua các group còn lại có thể thay đổi theo subset; tại m=7 chỉ số này không áp dụng.", "",
                  f"Đây là mô phỏng hiệu chỉnh annotation. Kết quả sửa toàn bộ concepts dùng diagnosis head {model_label} đã học "
                  "và không đồng nhất với oracle classifier M2 được huấn luyện riêng."])
    (output_dir / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


# Đọc preset/seeds, chạy frozen intervention từng seed rồi xuất báo cáo chung.
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--manifest_path", default=str(ROOT / "data/manifest.csv"))
    parser.add_argument("--output_dir")
    parser.add_argument("--diagnosis_head", choices=list(DIAGNOSIS_HEADS), default="linear")
    parser.add_argument("--diagnosis_lr", type=float)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--augmentation_preset", choices=["legacy_letterbox", "comparison"], default="legacy_letterbox")
    parser.add_argument("--concept_loss_weight", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    add_selection_argument(parser)
    args = parser.parse_args(argv)
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use at least three distinct seeds")
    defaults = [_default_paths(
        seed, args.augmentation_preset, args.concept_loss_weight, args.checkpoint_metric,
        args.diagnosis_head, args.diagnosis_lr, args.epochs, args.lr, args.weight_decay,
    ) for seed in args.seeds]
    # Intervention nằm trong cùng thư mục cấu hình với các JSON train/test.
    output = Path(args.output_dir or Path(defaults[0][1]).parent / "intervention").resolve()
    artifacts = [output / name for name in ("summary.json", "curve.csv", "per_concept.csv",
                 "full_intervention_cases.csv", "intervention_curve.png", "intervention_curve.svg", "report.md")]
    artifacts += [output / f"seed{seed}.json" for seed in args.seeds]
    if not args.overwrite and any(path.exists() for path in artifacts):
        raise FileExistsError("Intervention artifacts already exist; choose another --output_dir or use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed, (checkpoint, predictions) in zip(args.seeds, defaults):
        runs.append(run_m3_intervention(checkpoint, predictions, args.manifest_path,
                    str(output / f"seed{seed}.json"), args.overwrite))
    summary = summarize_interventions(runs)
    summary["source_paths"] = [str(output / f"seed{seed}.json") for seed in args.seeds]
    export_report(summary, runs, output)
    print(f"Saved three-seed tables and figures to {output}")
    return summary


if __name__ == "__main__":
    main()
