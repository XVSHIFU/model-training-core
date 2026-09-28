"""语义四分类与 TDP 兼容性验收指标。"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Callable

from upload_judge.features import response_format
from upload_judge.rules import SAFE_EXTENSIONS, SCRIPT_LIKE_EXTENSIONS
from upload_judge.schemas import UploadEvent

SUCCESS_LABELS = {"confirmed_upload_success", "likely_upload_success", "success"}
SUCCESS_PREDS = {"confirmed_upload_success", "likely_upload_success"}
ALL_PREDICTIONS = ["confirmed_upload_success", "likely_upload_success", "failed", "unknown"]


def _validate_lengths(*values: list) -> None:
    lengths = {len(value) for value in values}
    if len(lengths) > 1:
        raise ValueError(f"指标输入长度不一致: {sorted(lengths)}")


def normalize_gold(label: str) -> str:
    return "success" if label in SUCCESS_LABELS else "failed" if label == "failed" else label


def core_metrics(labels: list[str], predictions: list[str]) -> dict:
    _validate_lengths(labels, predictions)
    actual = [normalize_gold(value) for value in labels]
    total = len(actual)
    confusion = {gold: {pred: 0 for pred in ALL_PREDICTIONS} for gold in sorted(set(actual))}
    for gold, pred in zip(actual, predictions):
        confusion.setdefault(gold, {name: 0 for name in ALL_PREDICTIONS})[pred] += 1

    per_prediction: dict[str, dict[str, float | int]] = {}
    for prediction in ALL_PREDICTIONS:
        expected = "success" if prediction in SUCCESS_PREDS else "failed" if prediction == "failed" else None
        predicted_count = sum(pred == prediction for pred in predictions)
        true_positive = sum(gold == expected and pred == prediction for gold, pred in zip(actual, predictions)) if expected else 0
        support = sum(gold == expected for gold in actual) if expected else 0
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_prediction[prediction] = {
            "support": support,
            "predicted": predicted_count,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

    failed_total = sum(value == "failed" for value in actual)
    false_success = sum(gold == "failed" and pred in SUCCESS_PREDS for gold, pred in zip(actual, predictions))
    unknown = predictions.count("unknown")
    confirmed_precision = per_prediction["confirmed_upload_success"]["precision"]
    failed_precision = per_prediction["failed"]["precision"]
    return {
        "metric_mode": "tdp_v1",
        "total": total,
        "labels": dict(Counter(actual)),
        "predictions": dict(Counter(predictions)),
        "confusion": confusion,
        "per_class": per_prediction,
        "confirmed_upload_success_precision": confirmed_precision,
        "failed_precision": failed_precision,
        "unknown_rate": round(unknown / total, 6) if total else 0.0,
        "false_success_count": false_success,
        "false_success_rate": round(false_success / failed_total, 6) if failed_total else 0.0,
        "hard_gates": {
            "confirmed_precision_ge_0_90": bool(confirmed_precision >= 0.90),
            "failed_precision_ge_0_95": bool(failed_precision >= 0.95),
            "unknown_rate_le_0_30": bool((unknown / total if total else 1.0) <= 0.30),
            "false_success_rate_le_0_01": bool((false_success / failed_total if failed_total else 1.0) <= 0.01),
        },
    }


def semantic_confirmed_precision(gold: list[str], predictions: list[str]) -> float:
    _validate_lengths(gold, predictions)
    predicted = sum(value == "confirmed_upload_success" for value in predictions)
    correct = sum(
        expected == "confirmed_upload_success" and actual == "confirmed_upload_success"
        for expected, actual in zip(gold, predictions)
    )
    return correct / predicted if predicted else 0.0


def tdp_compatibility_metrics(
    events: list[UploadEvent],
    labels: list[str],
    predictions: list[str],
) -> dict:
    _validate_lengths(events, labels, predictions)
    result = core_metrics(labels, predictions)
    result["metric_mode"] = "tdp_v2_compatibility"
    return result


def semantic_metrics(labels: list[str], predictions: list[str]) -> dict:
    _validate_lengths(labels, predictions)
    invalid = sorted((set(labels) | set(predictions)) - set(ALL_PREDICTIONS))
    if invalid:
        raise ValueError(f"语义四分类出现未知标签: {invalid}")
    total = len(labels)
    confusion = {gold: {pred: 0 for pred in ALL_PREDICTIONS} for gold in ALL_PREDICTIONS}
    for gold, pred in zip(labels, predictions):
        confusion[gold][pred] += 1
    per_class = {}
    for name in ALL_PREDICTIONS:
        predicted = sum(value == name for value in predictions)
        support = sum(value == name for value in labels)
        true_positive = confusion[name][name]
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "support": support,
            "predicted": predicted,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    accuracy = sum(gold == pred for gold, pred in zip(labels, predictions)) / total if total else 0.0
    confirmed_precision = semantic_confirmed_precision(labels, predictions)
    failed_to_success = sum(
        gold == "failed" and pred in SUCCESS_PREDS for gold, pred in zip(labels, predictions)
    )
    return {
        "metric_mode": "semantic_gold_v2",
        "total": total,
        "labels": dict(Counter(labels)),
        "predictions": dict(Counter(predictions)),
        "confusion": confusion,
        "per_class": per_class,
        "accuracy_4": round(accuracy, 6),
        "semantic_confirmed_precision": round(confirmed_precision, 6),
        "failed_to_success": failed_to_success,
        "hard_gates": {
            "accuracy_4_ge_0_85": bool(accuracy >= 0.85),
            "semantic_confirmed_precision_ge_0_90": bool(confirmed_precision >= 0.90),
            "gold_failed_to_success_eq_0": failed_to_success == 0,
        },
    }


def build_evaluation(
    events: list[UploadEvent],
    labels: list[str],
    predictions: list[str],
    mode: str = "tdp_v2",
) -> dict:
    _validate_lengths(events, labels, predictions)
    if mode == "semantic":
        metric_fn = lambda ev, gold, pred: semantic_metrics(gold, pred)
    elif mode == "tdp_v1":
        metric_fn = lambda ev, gold, pred: core_metrics(gold, pred)
    elif mode == "tdp_v2":
        metric_fn = tdp_compatibility_metrics
    else:
        raise ValueError(f"未知评估模式: {mode}")
    result = metric_fn(events, labels, predictions)
    slices: dict[str, dict[str, dict]] = {}
    groupers: dict[str, Callable[[UploadEvent], str]] = {
        "response_format": response_format,
        "status_code": lambda event: str(event.response.status_code),
        "status_family": lambda event: f"{event.response.status_code // 100}xx" if event.response.status_code else "unknown",
        "extension": lambda event: (event.request.file_ext or "<none>").lower().strip("."),
        "extension_group": _extension_group,
        "template_cluster": _template_cluster,
    }
    for slice_name, grouper in groupers.items():
        indices: dict[str, list[int]] = defaultdict(list)
        for index, event in enumerate(events):
            indices[grouper(event)].append(index)
        ordered = sorted(indices.items(), key=lambda item: (-len(item[1]), item[0]))
        if slice_name == "template_cluster":
            ordered = [item for item in ordered if len(item[1]) >= 2][:100]
        slices[slice_name] = {}
        for key, values in ordered:
            sliced_events = [events[i] for i in values]
            sliced_labels = [labels[i] for i in values]
            sliced_predictions = [predictions[i] for i in values]
            slices[slice_name][key] = metric_fn(sliced_events, sliced_labels, sliced_predictions)
    result["slices"] = slices
    return result


def to_markdown(metrics: dict, title: str = "评估报告") -> str:
    if metrics.get("metric_mode") == "semantic_gold_v2":
        return _semantic_to_markdown(metrics, title)
    lines = [f"# {title}", "", "## 硬门槛", ""]
    lines.extend(
        [
            f"- 样本数：{metrics['total']}",
            f"- confirmed precision：{metrics['confirmed_upload_success_precision']:.4f}",
            f"- failed precision：{metrics['failed_precision']:.4f}",
            f"- unknown 占比：{metrics['unknown_rate']:.4f}",
            f"- 误判成功率：{metrics['false_success_rate']:.4f}（{metrics['false_success_count']} 条）",
            f"- 门槛结果：{'通过' if all(metrics['hard_gates'].values()) else '未通过'}",
            "",
            "## 混淆矩阵（TDP 二分类金标 × 四类预测）",
            "",
            "| actual \\ predicted | confirmed | likely | failed | unknown |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for actual, row in metrics["confusion"].items():
        lines.append(
            f"| {actual} | {row['confirmed_upload_success']} | {row['likely_upload_success']} | {row['failed']} | {row['unknown']} |"
        )
    lines.extend(["", "## 各预测类指标", "", "| class | predicted | precision | recall | F1 |", "|---|---:|---:|---:|---:|"])
    for name, values in metrics["per_class"].items():
        lines.append(f"| {name} | {values['predicted']} | {values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} |")
    for slice_name, groups in metrics.get("slices", {}).items():
        lines.extend(["", f"## 切片：{slice_name}", "", "| group | n | confirmed P | failed P | unknown | false-success |", "|---|---:|---:|---:|---:|---:|"])
        for group, values in list(groups.items())[:100]:
            lines.append(
                f"| {str(group).replace('|', '/')} | {values['total']} | {values['confirmed_upload_success_precision']:.4f} | "
                f"{values['failed_precision']:.4f} | {values['unknown_rate']:.4f} | {values['false_success_rate']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def _extension_group(event: UploadEvent) -> str:
    ext = (event.request.file_ext or "").lower().strip(".")
    if ext in SCRIPT_LIKE_EXTENSIONS:
        return "script"
    if ext in SAFE_EXTENSIONS:
        return "safe_extension"
    return "other"


def _template_cluster(event: UploadEvent) -> str:
    body = (event.response.body or "").lower()
    body = re.sub(r"\b[0-9a-f]{32,}\b", "<hash>", body)
    body = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", "<uuid>", body)
    body = re.sub(r"\b\d{4}-\d{1,2}-\d{1,2}(?:[t ]\d{1,2}:\d{2}:\d{2})?\b", "<time>", body)
    body = re.sub(r"\b\d+\b", "<num>", body)
    body = re.sub(r"\s+", " ", body).strip()[:4000]
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
    return f"{digest}:{body[:60]}"


def _semantic_to_markdown(metrics: dict, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "## 语义金标硬门槛",
        "",
        f"- 样本数：{metrics['total']}",
        f"- Accuracy_4：{metrics['accuracy_4']:.4f}",
        f"- semantic confirmed precision：{metrics['semantic_confirmed_precision']:.4f}",
        f"- gold failed→confirmed/likely：{metrics['failed_to_success']}",
        f"- 门槛结果：{'通过' if all(metrics['hard_gates'].values()) else '未通过'}",
        "",
        "## 四分类混淆矩阵",
        "",
        "| gold \\ predicted | confirmed | likely | failed | unknown |",
        "|---|---:|---:|---:|---:|",
    ]
    for actual, row in metrics["confusion"].items():
        lines.append(
            f"| {actual} | {row['confirmed_upload_success']} | {row['likely_upload_success']} | {row['failed']} | {row['unknown']} |"
        )
    lines.extend(["", "## 各语义类指标", "", "| class | support | predicted | precision | recall | F1 |", "|---|---:|---:|---:|---:|---:|"])
    for name, values in metrics["per_class"].items():
        lines.append(
            f"| {name} | {values['support']} | {values['predicted']} | {values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} |"
        )
    for slice_name, groups in metrics.get("slices", {}).items():
        lines.extend([
            "",
            f"## 切片：{slice_name}",
            "",
            "| group | n | Accuracy_4 | confirmed P | failed→success |",
            "|---|---:|---:|---:|---:|",
        ])
        for group, values in list(groups.items())[:100]:
            lines.append(
                f"| {str(group).replace('|', '/')} | {values['total']} | {values['accuracy_4']:.4f} | "
                f"{values['semantic_confirmed_precision']:.4f} | {values['failed_to_success']} |"
            )
    return "\n".join(lines) + "\n"
