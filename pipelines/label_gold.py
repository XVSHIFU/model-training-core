"""按冻结手册抽样、调用 Claude 标注、冻结和评分金标集。"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 同一工作区存在旧参考工程的同名 upload_judge 包；脚本必须优先使用 vNext 源码。
ROOT = Path(__file__).resolve().parents[1]
VNEXT_SRC = str(ROOT / "src")
if VNEXT_SRC not in sys.path:
    sys.path.insert(0, VNEXT_SRC)

from upload_judge.evidence import extract_evidence, has_failure_semantics, has_resource_evidence, has_success_semantics
from upload_judge.features import response_format
from upload_judge.judge import UploadJudge
from upload_judge.rules import SCRIPT_LIKE_EXTENSIONS, apply_rules
from upload_judge.metrics import semantic_metrics
from upload_judge.schemas import UploadEvent, Verdict

FINAL_TEST = ROOT / "data" / "processed" / "final_test.jsonl"
FINAL_TEST_V2 = ROOT / "data" / "processed" / "final_test_v2.jsonl"
GOLD_DIR = ROOT / "data" / "gold"
MANIFEST_DIR = ROOT / "data" / "manifests"
VERDICTS = ["confirmed_upload_success", "likely_upload_success", "failed", "unknown"]
SEED = 42

HANDBOOK = """固定标注手册（严格按优先级，只使用单次请求/响应）：
1. failed：明确失败/拒绝/拦截语义，如 success=false、上传或保存失败、类型/后缀不允许、权限不足、未授权、forbidden/access denied、WAF/病毒拦截、invalid file type/file too large、401/403 且有权限语义。
2. confirmed_upload_success：明确上传/保存成功语义（上传成功、保存成功、success=true、code=0、state=success、资源创建完成等），并伴随独立资源证据（URL、路径、Location、文件ID/附件ID或文件名回显）。
3. likely_upload_success：明确成功语义但缺独立资源证据；或多个中等强度证据一致且无失败证据。
4. unknown：仅 HTTP 200、裸 OK、空体、通用操作成功但无文件上下文、登录页/首页，或上传相关强成功与强失败证据冲突。
失败明确证据优先于通用成功字段。扩展名只作上下文；不要判断是否可执行、可访问或 WebShell 是否连接。"""

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "gold_verdict": {"type": "string", "enum": VERDICTS},
                    "reason": {"type": "string"},
                    "positive_evidence": {"type": "array", "items": {"type": "string"}},
                    "negative_evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["event_id", "gold_verdict", "reason", "positive_evidence", "negative_evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def sample_and_label(sample_size: int, output: Path, batch_size: int, model: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {output}")
    rows = list(_read_jsonl(FINAL_TEST))
    if len(rows) <= sample_size + 3000:
        raise RuntimeError(f"抽取 {sample_size} 条后 final_test 将低于 3000（当前 {len(rows)}）")
    selected = stratified_sample(rows, sample_size, SEED)
    annotations: dict[str, dict[str, Any]] = {}
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        checkpoint = json.loads(partial.read_text(encoding="utf-8"))
        selected_ids = {item["event_id"] for item in selected}
        for item in checkpoint:
            if item.get("event_id") in selected_ids and item.get("gold_verdict") in VERDICTS:
                annotations[item["event_id"]] = {
                    "event_id": item["event_id"],
                    "gold_verdict": item["gold_verdict"],
                    "reason": item.get("reason", ""),
                    "positive_evidence": item.get("positive_evidence", []),
                    "negative_evidence": item.get("negative_evidence", []),
                }
        print(f"从断点恢复: {len(annotations)}/{len(selected)}", flush=True)
    for offset in range(0, len(selected), batch_size):
        batch = [item for item in selected[offset: offset + batch_size] if item["event_id"] not in annotations]
        if not batch:
            continue
        labeled = call_claude(batch, model=model)
        expected = {item["event_id"] for item in batch}
        returned = {item["event_id"] for item in labeled}
        if expected != returned:
            labeled = _repair_event_ids(batch, labeled)
            returned = {item["event_id"] for item in labeled}
            if expected != returned:
                raise RuntimeError(f"Claude 返回 event_id 不匹配: missing={expected-returned}, extra={returned-expected}")
        annotations.update({item["event_id"]: item for item in labeled})
        checkpoint = [_merge_annotation(item, annotations[item["event_id"]]) for item in selected if item["event_id"] in annotations]
        output.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Claude 标注进度: {len(annotations)}/{len(selected)}", flush=True)

    candidates = [_merge_annotation(item, annotations[item["event_id"]]) for item in selected]
    counts = Counter(item["gold_verdict"] for item in candidates)
    output.write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    remaining = [item for item in rows if item["event_id"] not in annotations]
    before_hash = sha256_file(FINAL_TEST)
    _write_jsonl(FINAL_TEST, remaining)
    after_hash = sha256_file(FINAL_TEST)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "annotator": f"Claude CLI model={model}",
        "handbook_sha256": hashlib.sha256(HANDBOOK.encode("utf-8")).hexdigest(),
        "sample_size": len(candidates),
        "gold_verdict_counts": dict(counts),
        "candidate_path": str(output.resolve()),
        "candidate_sha256": sha256_file(output),
        "final_test_before": {"count": len(rows), "sha256": before_hash},
        "final_test_after": {"count": len(remaining), "sha256": after_hash},
        "sampled_event_ids": [item["event_id"] for item in candidates],
        "review_status": "pending_user_review",
        "available_status_families": dict(Counter(f"{int(item['response'].get('status_code', 0)) // 100}xx" for item in rows)),
        "sampling_strata": dict(Counter(item["sampling_stratum"] for item in candidates)),
        "coverage_note": "源 final_test 仅含 2xx，无法覆盖计划中期望的 4xx/5xx；已如实记录，未合成样本。",
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / "gold_sampling_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _update_dataset_manifest(len(remaining), after_hash, len(candidates), output)
    write_review_report(candidates, GOLD_DIR / "gold_review.md", output)
    return manifest


def stratified_sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    enriched = []
    for item in rows:
        event = UploadEvent.model_validate({key: value for key, value in item.items() if key != "label"})
        enriched.append((item, _sampling_stratum(event, item.get("label", ""))))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item, stratum in enriched:
        groups[stratum].append(item)
    for values in groups.values():
        rng.shuffle(values)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    # 先覆盖所有可用的格式/扩展名/规则结论组合，再轮询补齐，避免纯随机被 JSON 2xx 支配。
    ordered = sorted(groups, key=lambda key: (len(groups[key]), key))
    while len(selected) < size and ordered:
        next_round = []
        for key in ordered:
            values = groups[key]
            if values and len(selected) < size:
                item = values.pop()
                if item["event_id"] not in selected_ids:
                    selected.append(item)
                    selected_ids.add(item["event_id"])
            if values:
                next_round.append(key)
        ordered = next_round
    if len(selected) < size:
        remaining = [item for item, _ in enriched if item["event_id"] not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: size - len(selected)])
    rng.shuffle(selected)
    return selected[:size]


def sample_and_label_v2(sample_size: int, output: Path, batch_size: int, model: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {output}")
    if not FINAL_TEST_V2.exists():
        raise FileNotFoundError(FINAL_TEST_V2)
    rows = list(_read_jsonl(FINAL_TEST_V2))
    selected = stratified_sample(rows, sample_size, SEED)
    if len(rows) - len(selected) < 3000:
        raise RuntimeError(f"抽取 {len(selected)} 条后 final_test_v2 将低于 3000（当前 {len(rows)}）")
    annotations: dict[str, dict[str, Any]] = {}
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        checkpoint = json.loads(partial.read_text(encoding="utf-8"))
        selected_ids = {item["event_id"] for item in selected}
        annotations = {
            item["event_id"]: {
                "event_id": item["event_id"],
                "gold_verdict": item["gold_verdict"],
                "reason": item.get("reason", ""),
                "positive_evidence": item.get("positive_evidence", []),
                "negative_evidence": item.get("negative_evidence", []),
            }
            for item in checkpoint
            if item.get("event_id") in selected_ids and item.get("gold_verdict") in VERDICTS
        }
        print(f"从 v2 断点恢复: {len(annotations)}/{len(selected)}", flush=True)
    for offset in range(0, len(selected), batch_size):
        batch = [item for item in selected[offset: offset + batch_size] if item["event_id"] not in annotations]
        if not batch:
            continue
        labeled = _repair_event_ids(batch, call_claude(batch, model=model))
        expected = {item["event_id"] for item in batch}
        if {item["event_id"] for item in labeled} != expected:
            raise RuntimeError("Claude v2 标注 event_id 无法安全对齐")
        annotations.update({item["event_id"]: item for item in labeled})
        checkpoint = [_merge_annotation(item, annotations[item["event_id"]]) for item in selected if item["event_id"] in annotations]
        output.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Claude v2 标注进度: {len(annotations)}/{len(selected)}", flush=True)
    candidates = [_merge_annotation(item, annotations[item["event_id"]]) for item in selected]
    output.write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": 2,
        "seed": SEED,
        "annotator": f"Claude CLI model={model}",
        "handbook_sha256": hashlib.sha256(HANDBOOK.encode("utf-8")).hexdigest(),
        "sample_size": len(candidates),
        "gold_verdict_counts": dict(Counter(item["gold_verdict"] for item in candidates)),
        "candidate_path": str(output.resolve()),
        "candidate_sha256": sha256_file(output),
        "final_test_v2_unchanged": {"count": len(rows), "sha256": sha256_file(FINAL_TEST_V2)},
        "deduction_status": "pending_user_review_and_freeze",
        "review_status": "pending_user_review",
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / "gold_sampling_manifest_v2.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_review_report(candidates, GOLD_DIR / "gold_review_v2.md", output)
    return manifest


def call_claude(batch: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    executable = shutil.which("claude") or shutil.which("claude.cmd")
    if not executable:
        raise FileNotFoundError("未找到 Claude CLI")
    prompt_items = []
    for item in batch:
        prompt_items.append(
            {
                "event_id": item["event_id"],
                "request": {
                    "method": item["request"].get("method", ""),
                    "uri": item["request"].get("uri", ""),
                    "content_type": item["request"].get("content_type", ""),
                    "filename": item["request"].get("filename", ""),
                    "file_ext": item["request"].get("file_ext", ""),
                    "field_name": item["request"].get("field_name", ""),
                    "body_excerpt": str(item["request"].get("body_excerpt", ""))[:600],
                },
                "response": {
                    "status_code": item["response"].get("status_code", 0),
                    "headers": item["response"].get("headers", {}),
                    "content_type": item["response"].get("content_type", ""),
                    "body": str(item["response"].get("body", ""))[:6000],
                },
            }
        )
    prompt = (
        "你是文件上传成功性金标员。逐条独立判断，禁止参考输入中的 TDP 标签（此处未提供）。"
        "只返回符合 JSON Schema 的结构化结果，event_id 必须原样返回。\n\n"
        + HANDBOOK
        + "\n\n待标注事件：\n"
        + json.dumps(prompt_items, ensure_ascii=False)
    )
    command = [
        executable, "-p", "--safe-mode", "--no-session-persistence", "--tools", "",
        "--model", model, "--effort", "low", "--output-format", "json",
        "--json-schema", json.dumps(SCHEMA, ensure_ascii=False, separators=(",", ":")),
    ]
    last_error = ""
    for attempt in range(2):
        completed = subprocess.run(command, input=prompt, text=True, encoding="utf-8", capture_output=True, timeout=300)
        if completed.returncode == 0:
            envelope = json.loads(completed.stdout)
            structured = envelope.get("structured_output")
            if structured is None and isinstance(envelope.get("result"), str):
                structured = json.loads(envelope["result"])
            if isinstance(structured, dict) and isinstance(structured.get("items"), list):
                return structured["items"]
            last_error = f"缺少 structured_output: {completed.stdout[:500]}"
        else:
            last_error = completed.stderr[-1000:]
        print(f"Claude 批次重试 {attempt + 1}/2: {last_error}", file=sys.stderr, flush=True)
    raise RuntimeError(f"Claude 标注失败: {last_error}")


def freeze(candidates_path: Path, output: Path, review_note: str) -> dict[str, Any]:
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    counts = Counter(item.get("gold_verdict") for item in candidates)
    invalid = [item.get("event_id") for item in candidates if item.get("gold_verdict") not in VERDICTS]
    if invalid:
        raise ValueError(f"存在无效 gold_verdict: {invalid[:10]}")
    short = {name: counts[name] for name in VERDICTS if counts[name] < 30}
    if short:
        raise RuntimeError(f"各类至少 30 条未满足: {short}")
    is_v2 = "v2" in candidates_path.stem.lower() or "v2" in output.stem.lower()
    if is_v2:
        if not review_note.strip():
            raise RuntimeError("v2 冻结必须提供非空 --review-note")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "review_note": review_note,
        "source": str(candidates_path.resolve()),
        "source_sha256": sha256_file(candidates_path),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "counts": dict(counts),
    }
    if is_v2:
        if not FINAL_TEST_V2.exists():
            raise FileNotFoundError(FINAL_TEST_V2)
        final_rows = list(_read_jsonl(FINAL_TEST_V2))
        gold_ids = {item["event_id"] for item in candidates}
        remaining = [item for item in final_rows if item["event_id"] not in gold_ids]
        removed = len(final_rows) - len(remaining)
        if removed != len(gold_ids):
            raise RuntimeError(f"gold 扣除数量不一致: expected={len(gold_ids)}, removed={removed}")
        if len(remaining) < 3000:
            raise RuntimeError(f"gold 扣除后 final_test_v2 仅 {len(remaining)} 条")
        before_hash = sha256_file(FINAL_TEST_V2)
        _write_jsonl(FINAL_TEST_V2, remaining)
        manifest["final_test_v2_deduction"] = {
            "before_count": len(final_rows),
            "before_sha256": before_hash,
            "removed": removed,
            "after_count": len(remaining),
            "after_sha256": sha256_file(FINAL_TEST_V2),
        }
        final_manifest_path = MANIFEST_DIR / "final_window_v2_manifest.json"
        if final_manifest_path.exists():
            final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
            final_manifest["gold_deduction"] = manifest["final_test_v2_deduction"] | {
                "gold_path": str(output.resolve()),
                "gold_sha256": sha256_file(output),
            }
            final_manifest_path.write_text(json.dumps(final_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output.with_suffix(".manifest.json")).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sampling_manifest = MANIFEST_DIR / ("gold_sampling_manifest_v2.json" if is_v2 else "gold_sampling_manifest.json")
    if sampling_manifest.exists():
        current = json.loads(sampling_manifest.read_text(encoding="utf-8"))
        current["review_status"] = "frozen"
        current["review_note"] = review_note
        current["frozen_path"] = str(output.resolve())
        current["frozen_sha256"] = sha256_file(output)
        sampling_manifest.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def score(model_path: Path, gold_path: Path, output: Path | None = None) -> dict[str, Any]:
    rows = json.loads(gold_path.read_text(encoding="utf-8"))
    events = [UploadEvent.model_validate({key: value for key, value in item.items() if key not in {"label", "gold_verdict", "reason", "positive_evidence", "negative_evidence", "sampling_stratum", "review_status"}}) for item in rows]
    gold = [item["gold_verdict"] for item in rows]
    predicted = [result.verdict.value for result in UploadJudge(model_path=model_path).judge_batch(events)]
    result = semantic_metrics(gold, predicted)
    result["safety_passed"] = result["failed_to_success"] == 0
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_score_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def adjudicate(candidates_path: Path, output: Path, batch_size: int, model: str) -> dict[str, Any]:
    """对安全关键和证据不对齐项做不含原标签的独立二次标注。"""
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    selected = [item for item in candidates if item.get("gold_verdict") == "failed" or _semantic_mismatch(item)]
    second_labels: dict[str, dict[str, Any]] = {}
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        checkpoint = json.loads(partial.read_text(encoding="utf-8"))
        second_labels = {item["event_id"]: item["second_annotation"] for item in checkpoint if item.get("second_annotation")}
        print(f"从复标断点恢复: {len(second_labels)}/{len(selected)}", flush=True)
    for offset in range(0, len(selected), batch_size):
        batch = [item for item in selected[offset: offset + batch_size] if item["event_id"] not in second_labels]
        if not batch:
            continue
        # call_claude 只读取 request/response，不会把原 gold_verdict 传入 prompt。
        labeled = _repair_event_ids(batch, call_claude(batch, model=model))
        expected = {item["event_id"] for item in batch}
        if {item["event_id"] for item in labeled} != expected:
            raise RuntimeError("二次复标 event_id 无法安全对齐")
        second_labels.update({item["event_id"]: item for item in labeled})
        checkpoint = [
            {"event_id": item["event_id"], "first_verdict": item["gold_verdict"], "second_annotation": second_labels[item["event_id"]]}
            for item in selected if item["event_id"] in second_labels
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Claude 独立复标进度: {len(second_labels)}/{len(selected)}", flush=True)
    items = []
    for item in selected:
        second = second_labels[item["event_id"]]
        items.append(
            {
                "event_id": item["event_id"],
                "first_verdict": item["gold_verdict"],
                "second_verdict": second["gold_verdict"],
                "agreed": item["gold_verdict"] == second["gold_verdict"],
                "first_reason": item.get("reason", ""),
                "second_reason": second.get("reason", ""),
                "filename": item["request"].get("filename", ""),
                "response_excerpt": " ".join(str(item["response"].get("body", "")).split())[:500],
            }
        )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "selection": "all first-pass failed plus deterministic evidence mismatches",
        "selected": len(items),
        "agreed": sum(item["agreed"] for item in items),
        "disagreed": sum(not item["agreed"] for item in items),
        "transition_counts": dict(Counter(f"{item['first_verdict']}->{item['second_verdict']}" for item in items)),
        "items": items,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    output.with_suffix(".md").write_text(_adjudication_markdown(result), encoding="utf-8")
    return result


def augment_failed_coverage(
    candidates_path: Path,
    output: Path,
    sample_size: int,
    batch_size: int,
    model: str,
) -> dict[str, Any]:
    """盲抽一批明确失败候选；无论 Claude 输出何类都整体加入 gold，避免按结果挑样。"""
    if output.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {output}")
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    final_rows = list(_read_jsonl(FINAL_TEST))
    ranked = sorted(
        (item for item in final_rows if item.get("label") == "failed" and _failure_candidate_score(item) > 0),
        key=lambda item: (-_failure_candidate_score(item), item["event_id"]),
    )
    selected: list[dict[str, Any]] = []
    templates: set[str] = set()
    for item in ranked:
        body = re.sub(r"\b\d+\b", "<num>", str(item["response"].get("body", "")).lower())
        template = hashlib.sha256((item["request"].get("uri", "") + "|" + body).encode("utf-8")).hexdigest()
        if template in templates:
            continue
        templates.add(template)
        selected.append(item)
        if len(selected) == sample_size:
            break
    if len(selected) < sample_size:
        raise RuntimeError(f"仅找到 {len(selected)} 条去模板的强失败候选")
    annotations: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(selected), batch_size):
        batch = selected[offset: offset + batch_size]
        labeled = _repair_event_ids(batch, call_claude(batch, model=model))
        expected = {item["event_id"] for item in batch}
        if {item["event_id"] for item in labeled} != expected:
            raise RuntimeError("补标 event_id 无法安全对齐")
        annotations.update({item["event_id"]: item for item in labeled})
        print(f"Claude 强失败补标进度: {len(annotations)}/{len(selected)}", flush=True)
    augmented = candidates + [_merge_annotation(item, annotations[item["event_id"]]) for item in selected]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(augmented, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    selected_ids = {item["event_id"] for item in selected}
    remaining = [item for item in final_rows if item["event_id"] not in selected_ids]
    if len(remaining) < 3000:
        raise RuntimeError("补标后 final_test 将低于 3000")
    _write_jsonl(FINAL_TEST, remaining)
    final_hash = sha256_file(FINAL_TEST)
    _update_dataset_manifest(len(remaining), final_hash, len(augmented), output)
    write_review_report(augmented, GOLD_DIR / "gold_review_augmented.md", output)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_candidates": str(candidates_path.resolve()),
        "source_count": len(candidates),
        "augmentation_count": len(selected),
        "augmentation_verdicts": dict(Counter(annotations[item["event_id"]]["gold_verdict"] for item in selected)),
        "output": str(output.resolve()),
        "output_count": len(augmented),
        "output_sha256": sha256_file(output),
        "final_test_count": len(remaining),
        "final_test_sha256": final_hash,
        "sampled_event_ids": sorted(selected_ids),
    }
    (MANIFEST_DIR / "gold_augmentation_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _failure_candidate_score(item: dict[str, Any]) -> int:
    uri = str(item["request"].get("uri", ""))
    if "FalseAlarmDetection" in uri or "/3dpModelMall/" in uri:
        return 0
    body = str(item["response"].get("body", ""))
    score = 0
    if re.search(r"上传失败|保存失败|解析文件失败|invalid file|file too large|forbidden|unauthorized", body, re.I):
        score += 5
    if re.search(r'["\']success["\']\s*:\s*false', body, re.I):
        score += 3
    if re.search(r'["\'](?:code|status)["\']\s*:\s*["\']?(?:401|403|404|500)\b', body, re.I):
        score += 2
    if re.search(r"失败|不支持|拒绝|未登录|异常|无法|error|not found", body, re.I):
        score += 1
    return score


def _semantic_mismatch(item: dict[str, Any]) -> bool:
    ignored = {"source_tdp_label", "sampling_stratum", "gold_verdict", "reason", "positive_evidence", "negative_evidence", "review_status"}
    event = UploadEvent.model_validate({key: value for key, value in item.items() if key not in ignored})
    positive, negative = extract_evidence(event)
    has_success = has_success_semantics(positive)
    has_resource = has_resource_evidence(positive)
    has_failure = has_failure_semantics(negative)
    verdict = item["gold_verdict"]
    return (
        (verdict == "confirmed_upload_success" and not (has_success and has_resource))
        or (verdict == "likely_upload_success" and (has_failure or not has_success))
        or (verdict == "failed" and not has_failure)
        or (verdict == "unknown" and has_success and has_resource and not has_failure)
    )


def _adjudication_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Gold 独立复标差异",
        "",
        f"- 复标样本：{result['selected']}",
        f"- 一致：{result['agreed']}",
        f"- 不一致：{result['disagreed']}",
        "",
        "## 转移统计",
        "",
    ]
    lines.extend(f"- {name}: {count}" for name, count in sorted(result["transition_counts"].items()))
    lines.extend(["", "## 需用户裁决的差异", ""])
    for item in result["items"]:
        if item["agreed"]:
            continue
        response_excerpt = item["response_excerpt"].replace("`", "'")
        lines.extend(
            [
                f"### {item['event_id']}: {item['first_verdict']} → {item['second_verdict']}",
                "",
                f"- 文件：`{item['filename']}`",
                f"- 首标理由：{item['first_reason']}",
                f"- 复标理由：{item['second_reason']}",
                f"- 响应：`{response_excerpt}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _sampling_stratum(event: UploadEvent, source_label: str) -> str:
    rule = apply_rules(event)
    if rule:
        baseline = rule.verdict.value
    else:
        positive, _ = extract_evidence(event)
        baseline = "likely_upload_success" if has_success_semantics(positive) else "failed" if source_label == "failed" else "unknown"
        if has_success_semantics(positive) and has_resource_evidence(positive):
            baseline = "confirmed_upload_success"
    ext = (event.request.file_ext or "").lower().strip(".")
    ext_group = "script" if ext in SCRIPT_LIKE_EXTENSIONS else "other"
    return f"{baseline}|{source_label}|{response_format(event)}|{event.response.status_code // 100}xx|{ext_group}"


def _merge_annotation(item: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(item, ensure_ascii=False))
    event = UploadEvent.model_validate({key: value for key, value in item.items() if key != "label"})
    result["source_tdp_label"] = result.pop("label", "")
    result["sampling_stratum"] = _sampling_stratum(event, result["source_tdp_label"])
    result.update(annotation)
    result["review_status"] = "unreviewed"
    return result


def _repair_event_ids(batch: list[dict[str, Any]], labeled: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """只修复可一一确定的抄写误差，避免把标签配到错误事件。"""
    expected = {item["event_id"] for item in batch}
    returned = {item["event_id"] for item in labeled}
    missing, extra = list(expected - returned), list(returned - expected)
    if len(missing) != len(extra):
        return labeled
    mapping: dict[str, str] = {}
    unused = set(missing)
    for wrong in extra:
        ranked = sorted(
            ((difflib.SequenceMatcher(None, wrong, candidate).ratio(), candidate) for candidate in unused),
            reverse=True,
        )
        if not ranked or ranked[0][0] < 0.85 or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
            return labeled
        mapping[wrong] = ranked[0][1]
        unused.remove(ranked[0][1])
    repaired = []
    for item in labeled:
        value = dict(item)
        value["event_id"] = mapping.get(value["event_id"], value["event_id"])
        repaired.append(value)
    return repaired


def _update_dataset_manifest(final_count: int, final_hash: str, gold_count: int, candidates_path: Path) -> None:
    path = MANIFEST_DIR / "dataset_manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["outputs"]["final_test"]["count"] = final_count
    manifest["outputs"]["final_test"]["sha256"] = final_hash
    manifest["gold_deduction"] = {
        "count": gold_count,
        "candidates_path": str(candidates_path.resolve()),
        "candidates_sha256": sha256_file(candidates_path),
        "final_test_count_after_deduction": final_count,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_review_report(candidates: list[dict[str, Any]], output: Path, candidates_path: Path) -> None:
    gold_counts = Counter(item["gold_verdict"] for item in candidates)
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for item in candidates:
        matrix[item.get("source_tdp_label", "")][item["gold_verdict"]] += 1
    lines = [
        "# Gold candidates 用户复核单",
        "",
        f"请抽查下列分层样本；如需修正，直接修改 `{candidates_path.name}` 中对应条目的 `gold_verdict`、`reason` 和证据字段，然后告知执行冻结。",
        "",
        "## 分布",
        "",
    ]
    lines.extend(f"- {name}: {gold_counts[name]}" for name in VERDICTS)
    lines.extend(["", "## TDP 标签 × Claude 金标", "", "| TDP \\ gold | confirmed | likely | failed | unknown |", "|---|---:|---:|---:|---:|"])
    for source in sorted(matrix):
        row = matrix[source]
        lines.append(f"| {source} | {row['confirmed_upload_success']} | {row['likely_upload_success']} | {row['failed']} | {row['unknown']} |")
    lines.extend(["", "## 分层抽查样本（每类最多 8 条）", ""])
    for verdict in VERDICTS:
        lines.extend([f"### {verdict}", ""])
        values = [item for item in candidates if item["gold_verdict"] == verdict][:8]
        for item in values:
            response = " ".join(str(item["response"].get("body", "")).split())[:240]
            response = response.replace("`", "'")
            lines.extend(
                [
                    f"- `{item['event_id']}` — `{item['request'].get('filename', '')}` / HTTP {item['response'].get('status_code', 0)}",
                    f"  - 响应：`{response}`",
                    f"  - 理由：{item.get('reason', '')}",
                ]
            )
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _score_markdown(result: dict[str, Any]) -> str:
    lines = ["# 金标四分类报告", "", f"- total: {result['total']}", f"- Accuracy_4: {result['accuracy_4']:.4f}", f"- Semantic_Confirmed_Precision: {result['semantic_confirmed_precision']:.4f}", f"- failed→confirmed/likely: {result['failed_to_success']}", f"- safety: {'通过' if result['safety_passed'] else '未通过'}", "", "## 混淆矩阵", "", "| actual \\ predicted | confirmed | likely | failed | unknown |", "|---|---:|---:|---:|---:|"]
    for actual in VERDICTS:
        row = result["confusion"][actual]
        lines.append(f"| {actual} | {row['confirmed_upload_success']} | {row['likely_upload_success']} | {row['failed']} | {row['unknown']} |")
    return "\n".join(lines) + "\n"


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--sample", type=int)
    actions.add_argument("--freeze", type=Path)
    actions.add_argument("--adjudicate", type=Path)
    actions.add_argument("--augment", type=Path)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--score", action="store_true")
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--augment-size", type=int, default=20)
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--review-note", default="")
    args = parser.parse_args()
    if args.sample:
        output = args.out or GOLD_DIR / "gold_candidates.json"
        manifest = (
            sample_and_label_v2(args.sample, output, args.batch_size, args.model)
            if "v2" in output.stem.lower()
            else sample_and_label(args.sample, output, args.batch_size, args.model)
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.freeze:
        output = args.out or GOLD_DIR / "gold_blind_v1.json"
        print(json.dumps(freeze(args.freeze, output, args.review_note), ensure_ascii=False, indent=2))
        return 0
    if args.adjudicate:
        output = args.out or GOLD_DIR / "gold_adjudication.json"
        result = adjudicate(args.adjudicate, output, args.batch_size, args.model)
        print(json.dumps({key: result[key] for key in ("selected", "agreed", "disagreed", "transition_counts")}, ensure_ascii=False, indent=2))
        return 0
    if args.augment:
        output = args.out or GOLD_DIR / "gold_candidates_augmented.json"
        result = augment_failed_coverage(args.augment, output, args.augment_size, args.batch_size, args.model)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if len(args.paths) != 2:
        parser.error("--check/--score 需要: <model> <gold_json>")
    result = score(Path(args.paths[0]), Path(args.paths[1]), args.out if args.score else None)
    return 0 if (not args.check or result["safety_passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
