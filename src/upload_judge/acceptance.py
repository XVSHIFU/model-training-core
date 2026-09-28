"""v2 语义金标 + TDP 兼容指标的一键验收。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from upload_judge.judge import UploadJudge
from upload_judge.metrics import build_evaluation, to_markdown
from upload_judge.schemas import UploadEvent


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else [value]


def _load_labeled(path: Path) -> tuple[list[UploadEvent], list[str]]:
    events, labels = [], []
    for raw in _load_records(path):
        item = dict(raw)
        label = str(item.pop("label", item.pop("gold_verdict", "")))
        item.pop("raw", None)
        if not label:
            raise ValueError(f"缺少标签: {item.get('event_id')}")
        events.append(UploadEvent.model_validate(item))
        labels.append(label)
    return events, labels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_formal_inputs(gold_events, gold_labels, tdp_events, tdp_labels) -> dict[str, Any]:
    gold_counts = Counter(gold_labels)
    required_gold = {"confirmed_upload_success", "likely_upload_success", "failed", "unknown"}
    tdp_values = set(tdp_labels)
    gold_ids = {event.event_id for event in gold_events}
    tdp_ids = {event.event_id for event in tdp_events}
    checks = {
        "tdp_total_ge_3000": len(tdp_events) >= 3000,
        "tdp_labels_binary": tdp_values <= {"confirmed_upload_success", "failed"},
        "gold_all_four_classes": set(gold_counts) == required_gold,
        "gold_each_class_ge_30": all(gold_counts.get(label, 0) >= 30 for label in required_gold),
        "gold_deducted_from_tdp": not (gold_ids & tdp_ids),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "tdp_total": len(tdp_events),
        "tdp_labels": dict(Counter(tdp_labels)),
        "gold_total": len(gold_events),
        "gold_labels": dict(gold_counts),
        "event_id_overlap": len(gold_ids & tdp_ids),
    }


def _parity(judge: UploadJudge, events, batch_dicts, limit: int | None = None) -> dict[str, Any]:
    selected_events = events if limit is None else events[:limit]
    selected_batch = batch_dicts if limit is None else batch_dicts[:limit]
    singles = [judge.judge(event).to_dict() for event in selected_events]
    mismatches = [
        selected_events[index].event_id
        for index, (single, batch) in enumerate(zip(singles, selected_batch))
        if single != batch
    ]
    return {
        "scope": "full" if limit is None else f"first_{limit}",
        "total": len(selected_events),
        "mismatch_count": len(mismatches),
        "mismatch_event_ids": mismatches[:100],
        "passed": not mismatches,
    }


def run_acceptance(
    model_path: Path,
    gold_path: Path,
    tdp_path: Path,
    out_path: Path,
    json_out: Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    for path in (model_path, gold_path, tdp_path):
        if not path.exists():
            raise FileNotFoundError(path)
    sentinel = json_out.parent / ".acceptance_v2_consumed.json"
    if not smoke:
        if sentinel.exists():
            raise RuntimeError(f"正式验收已消费，拒绝再次运行: {sentinel}")
        if out_path.exists() or json_out.exists():
            raise RuntimeError("正式验收输出已存在，拒绝覆盖")

    gold_events, gold_labels = _load_labeled(gold_path)
    tdp_events, tdp_labels = _load_labeled(tdp_path)
    integrity = _validate_formal_inputs(gold_events, gold_labels, tdp_events, tdp_labels)
    if not smoke and not integrity["passed"]:
        raise RuntimeError(f"正式验收输入清单未通过: {integrity}")

    run_id = hashlib.sha256(
        f"{_sha256(model_path)}:{_sha256(gold_path)}:{_sha256(tdp_path)}".encode("utf-8")
    ).hexdigest()[:16]
    if not smoke:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(
            json.dumps({
                "status": "running",
                "run_id": run_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "model_sha256": _sha256(model_path),
                "gold_sha256": _sha256(gold_path),
                "tdp_sha256": _sha256(tdp_path),
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    try:
        judge = UploadJudge(model_path=model_path)
        gold_results = judge.judge_batch(gold_events)
        tdp_results = judge.judge_batch(tdp_events)
        gold_predictions = [item.verdict.value for item in gold_results]
        tdp_predictions = [item.verdict.value for item in tdp_results]
        semantic = build_evaluation(gold_events, gold_labels, gold_predictions, mode="semantic")
        tdp = build_evaluation(tdp_events, tdp_labels, tdp_predictions, mode="tdp_v2")
        gold_dicts = [item.to_dict() for item in gold_results]
        tdp_dicts = [item.to_dict() for item in tdp_results]
        parity_limit = 50 if smoke else None
        parity = {
            "gold": _parity(judge, gold_events, gold_dicts, parity_limit),
            "tdp": _parity(judge, tdp_events, tdp_dicts, parity_limit),
        }
        gates = {
            "semantic_gold": semantic["hard_gates"],
            "tdp_compatibility": tdp["hard_gates"],
            "single_batch_parity": {
                "gold_zero_mismatch": parity["gold"]["passed"],
                "tdp_zero_mismatch": parity["tdp"]["passed"],
            },
        }
        all_passed = all(value for group in gates.values() for value in group.values())
        report = {
            "version": 2,
            "mode": "smoke" if smoke else "formal_one_shot",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if all_passed else "failed",
            "model": {"path": str(model_path.resolve()), "sha256": _sha256(model_path)},
            "inputs": {
                "gold": {"path": str(gold_path.resolve()), "sha256": _sha256(gold_path)},
                "tdp": {"path": str(tdp_path.resolve()), "sha256": _sha256(tdp_path)},
            },
            "input_integrity": integrity,
            "semantic_gold": semantic,
            "tdp_compatibility": tdp,
            "single_batch_parity": parity,
            "gates": gates,
            "all_gates_passed": all_passed,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        out_path.write_text(_acceptance_markdown(report), encoding="utf-8")
        if not smoke:
            sentinel.write_text(
                json.dumps({
                    "status": "completed",
                    "run_id": run_id,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "all_gates_passed": all_passed,
                    "report_sha256": _sha256(json_out),
                }, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return report
    except Exception as exc:
        if not smoke and sentinel.exists():
            sentinel.write_text(
                json.dumps({
                    "status": "failed_consumed",
                    "run_id": run_id,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                }, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        raise


def _acceptance_markdown(report: dict[str, Any]) -> str:
    semantic = report["semantic_gold"]
    tdp = report["tdp_compatibility"]
    parity = report["single_batch_parity"]
    lines = [
        "# 文件上传成功性研判 v2 验收报告",
        "",
        f"- 模式：{report['mode']}",
        f"- run_id：`{report['run_id']}`",
        f"- 总结：{'通过' if report['all_gates_passed'] else '未通过'}",
        f"- Gold Accuracy_4：{semantic['accuracy_4']:.4f}",
        f"- Gold Semantic Confirmed Precision：{semantic['semantic_confirmed_precision']:.4f}",
        f"- Gold failed→confirmed/likely：{semantic['failed_to_success']}",
        f"- TDP false-success：{tdp['false_success_rate']:.4f}（{tdp['false_success_count']} 条）",
        f"- TDP confirmed precision：{tdp['confirmed_upload_success_precision']:.4f}",
        f"- TDP failed precision：{tdp['failed_precision']:.4f}",
        f"- TDP unknown rate：{tdp['unknown_rate']:.4f}",
        f"- single/batch parity：gold={parity['gold']['mismatch_count']}，tdp={parity['tdp']['mismatch_count']}",
    ]
    lines.extend(["", to_markdown(semantic, "语义金标四分类").rstrip(), "", to_markdown(tdp, "TDP v2 兼容指标").rstrip(), ""])
    return "\n".join(lines)
