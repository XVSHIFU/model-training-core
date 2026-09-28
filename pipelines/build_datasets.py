"""构建固定 train/val/final_test 三分区及可回溯 manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
VNEXT_SRC = str(ROOT / "src")
if VNEXT_SRC not in sys.path:
    sys.path.insert(0, VNEXT_SRC)

from upload_judge.convert import SENSITIVE_HEADER_NAMES, iter_converted, normalize_uri
from upload_judge.schemas import UploadEvent

DEFAULT_SOURCE = ROOT.parent / "file_upload_success_detected (2)"
PROCESSED = ROOT / "data" / "processed"
MANIFESTS = ROOT / "data" / "manifests"
SEED = 42
GOLD_V1 = ROOT / "data" / "gold" / "gold_blind_v1.json"

PHASE_FILES = [
    "train_phase1_10820.json",
    "test_phase1_2705.json",
    "train_phase2_16000.json",
    "test_phase2_4000.json",
]
RAW_FILES = [
    ("tdp_upload_success_detail.jsonl", "confirmed_upload_success"),
    ("tdp_upload_failed_detail.jsonl", "failed"),
]
FINAL_FILE = "test_v3_10000.json"

_HEX = re.compile(r"\b[0-9a-fA-F]{16,}\b")
_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_TIMESTAMP = re.compile(r"\b(?:1[5-9]\d{8}|2\d{9}|\d{4}-\d{1,2}-\d{1,2}(?:[T ]\d{1,2}:\d{2}:\d{2})?)\b")
_NUMBER = re.compile(r"\b\d+\b")


def exact_fingerprint(item: dict[str, Any]) -> str:
    req, resp = item["request"], item["response"]
    value = "|".join(
        [
            str(req.get("method", "")).upper(),
            normalize_uri(req.get("uri", "")),
            str(resp.get("status_code", 0)),
            str(resp.get("content_type", "")).lower().split(";", 1)[0].strip(),
            str(resp.get("body", "")),
        ]
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def template_fingerprint(item: dict[str, Any]) -> str:
    req, resp = item["request"], item["response"]
    body = str(resp.get("body", ""))
    body = _UUID.sub("<uuid>", body)
    body = _HEX.sub("<hex>", body)
    body = _TIMESTAMP.sub("<timestamp>", body)
    body = _NUMBER.sub("<num>", body)
    value = "|".join(
        [
            str(req.get("method", "")).upper(),
            normalize_uri(req.get("uri", "")),
            str(resp.get("status_code", 0)),
            str(resp.get("content_type", "")).lower().split(";", 1)[0].strip(),
            body,
        ]
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def standardize(item: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(item, ensure_ascii=False))
    value["request"]["uri"] = normalize_uri(value["request"].get("uri", ""))
    for side in ("request", "response"):
        headers = value[side].get("headers") or {}
        if isinstance(headers, str):
            headers = _parse_header_text(headers)
        redacted = {}
        for name, header_value in headers.items():
            lowered = str(name).lower()
            redacted[str(name)] = "<redacted>" if lowered in SENSITIVE_HEADER_NAMES or "token" in lowered or "secret" in lowered else str(header_value)
        value[side]["headers"] = redacted
    value["request"]["body_excerpt"] = str(value["request"].get("body_excerpt") or "")[:500]
    value["response"]["body"] = str(value["response"].get("body") or "")[:20000]
    value["label"] = _normalize_label(str(value.get("label", "")))
    UploadEvent.model_validate({key: entry for key, entry in value.items() if key != "label"})
    return value


def build(source_root: Path, include_raw: bool = True, seed: int = SEED) -> dict[str, Any]:
    sample_root, raw_root = source_root / "data" / "samples", source_root / "data" / "raw"
    source_paths = [sample_root / name for name in PHASE_FILES] + [sample_root / FINAL_FILE]
    if include_raw:
        source_paths.extend(raw_root / name for name, _ in RAW_FILES)
    missing = [str(path) for path in source_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少源文件: " + ", ".join(missing))

    training_candidates: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for name in PHASE_FILES:
        path = sample_root / name
        records = json.loads(path.read_text(encoding="utf-8"))
        normalized = [standardize(item) for item in records]
        training_candidates.extend(normalized)
        source_counts[name] = len(normalized)
    if include_raw:
        for name, label in RAW_FILES:
            path = raw_root / name
            count = 0
            for item in iter_converted(path, label):
                training_candidates.append(standardize(item))
                count += 1
            source_counts[name] = count

    training_unique, training_stats, conflicts = _dedupe(training_candidates, template_cap=1, exclude_template_conflicts=True)
    train, val = _stratified_split(training_unique, val_ratio=0.10, seed=seed)

    final_raw = json.loads((sample_root / FINAL_FILE).read_text(encoding="utf-8"))
    final_candidates = [standardize(item) for item in final_raw]
    blocked_exact = {exact_fingerprint(item) for item in train + val}
    final_candidates = [item for item in final_candidates if exact_fingerprint(item) not in blocked_exact]
    # 独立窗需同时满足模板降重与 final_test >= 3000。训练池每模板留 1 条；
    # final_test 按标签每模板最多留 50 条（原始最大模板 1552 条），限制头部模板
    # 支配，并为后续抽走约 300 条 gold 预留 final_test >= 3000 的空间。
    final_test, final_stats, final_conflicts = _dedupe(final_candidates, template_cap=50, exclude_template_conflicts=False)
    conflicts.extend(final_conflicts)
    if len(final_test) < 3300:
        raise RuntimeError(f"final_test 仅 {len(final_test)} 条，未给 300 条 gold 预留后仍满足 3000 的空间")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    outputs = {"train": train, "val": val, "final_test": final_test}
    output_meta = {}
    for name, rows in outputs.items():
        target = PROCESSED / f"{name}.jsonl"
        _write_jsonl(target, rows)
        output_meta[name] = {
            "path": str(target),
            "sha256": sha256_file(target),
            "count": len(rows),
            "labels": dict(Counter(item["label"] for item in rows)),
            "response_formats": dict(Counter(_response_format(item) for item in rows)),
            "status_families": dict(Counter(_status_family(item) for item in rows)),
            "extension_groups": dict(Counter(_extension_group(item) for item in rows)),
        }

    leak = leakage_report(outputs)
    if any(leak["exact_overlaps"].values()):
        raise RuntimeError(f"存在精确跨分区重复: {leak['exact_overlaps']}")
    (MANIFESTS / "leakage_report.json").write_text(json.dumps(leak, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(MANIFESTS / "label_conflicts.jsonl", conflicts)
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "source_root": str(source_root.resolve()),
        "include_raw": include_raw,
        "source_files": {
            path.name: {"path": str(path), "sha256": sha256_file(path), "count": source_counts.get(path.name, len(final_raw) if path.name == FINAL_FILE else None)}
            for path in source_paths
        },
        "cleaning": {
            "encoding": "utf-8",
            "exact_fingerprint": "sha256(method|norm_uri|status|content_type|body)",
            "template_normalization": ["uuid", "hex>=16", "unix/date timestamp", "numbers"],
            "template_caps": {"train_val_pool": 1, "final_test_per_label": 50},
            "template_conflicts": {"train_val_pool": "excluded", "final_test": "retained_per_label_and_reported"},
            "sensitive_headers_redacted": True,
            "training": training_stats,
            "final_test": final_stats,
            "conflicts_excluded": len(conflicts),
        },
        "outputs": output_meta,
        "leakage": leak,
        "final_test_policy": "07-04/05 独立时间窗；尚未运行模型评估；后续抽取 gold 时需从此文件扣除",
    }
    (MANIFESTS / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def leakage_report(partitions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    names = list(partitions)
    exact_sets = {name: {exact_fingerprint(item) for item in rows} for name, rows in partitions.items()}
    template_sets = {name: {template_fingerprint(item) for item in rows} for name, rows in partitions.items()}
    exact, template = {}, {}
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            key = f"{left}__{right}"
            exact[key] = len(exact_sets[left] & exact_sets[right])
            template[key] = len(template_sets[left] & template_sets[right])
    return {"exact_overlaps": exact, "template_overlaps_report_only": template, "exact_requirement_passed": all(value == 0 for value in exact.values())}


def report_existing() -> dict[str, Any]:
    partitions = {name: list(_read_jsonl(PROCESSED / f"{name}.jsonl")) for name in ("train", "val", "final_test")}
    report = leakage_report(partitions)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def split_gold_tune_regress(
    rows: list[dict[str, Any]],
    tune_size: int = 220,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按语义类、模板簇做确定性拆分，确保同模板不跨 tune/regress。"""
    if not 0 < tune_size < len(rows):
        raise ValueError("tune_size 必须大于 0 且小于金标总数")
    label_counts = Counter(str(item.get("gold_verdict", "")) for item in rows)
    if not label_counts or "" in label_counts:
        raise ValueError("金标记录缺少 gold_verdict")
    raw_targets = {label: count * tune_size / len(rows) for label, count in label_counts.items()}
    targets = {label: int(value) for label, value in raw_targets.items()}
    remainder = tune_size - sum(targets.values())
    for label in sorted(label_counts, key=lambda name: (-(raw_targets[name] - targets[name]), name))[:remainder]:
        targets[label] += 1

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        grouped[(str(item["gold_verdict"]), template_fingerprint(item))].append(item)
    selected_templates: set[tuple[str, str]] = set()
    for label, target in sorted(targets.items()):
        clusters = [
            (fingerprint, values)
            for (cluster_label, fingerprint), values in grouped.items()
            if cluster_label == label
        ]
        clusters.sort(key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode("utf-8")).hexdigest())
        choices: dict[int, tuple[str, ...]] = {0: ()}
        for fingerprint, values in clusters:
            size = len(values)
            for subtotal, chosen in sorted(list(choices.items()), reverse=True):
                new_total = subtotal + size
                if new_total <= target and new_total not in choices:
                    choices[new_total] = (*chosen, fingerprint)
        if target not in choices:
            raise RuntimeError(f"{label} 无法按模板簇精确拆出 {target} 条")
        selected_templates.update((label, fingerprint) for fingerprint in choices[target])

    tune, regress = [], []
    for item in rows:
        key = (str(item["gold_verdict"]), template_fingerprint(item))
        (tune if key in selected_templates else regress).append(item)
    order_key = lambda item: hashlib.sha256(f"{seed}:{item['event_id']}".encode("utf-8")).hexdigest()
    tune.sort(key=order_key)
    regress.sort(key=order_key)
    if len(tune) != tune_size:
        raise RuntimeError(f"金标调优拆分数量异常: {len(tune)} != {tune_size}")
    if {template_fingerprint(item) for item in tune} & {template_fingerprint(item) for item in regress}:
        raise RuntimeError("gold_tune 与 gold_regress 存在模板泄漏")
    return tune, regress


def split_gold_for_v2(gold_path: Path = GOLD_V1, tune_size: int = 220, seed: int = SEED) -> dict[str, Any]:
    """把旧金标拆为决策层调优集和只读回归集；两者均不进入模型训练。"""
    if not gold_path.exists():
        raise FileNotFoundError(gold_path)
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    if not isinstance(gold, list):
        raise ValueError("gold_blind_v1 必须是 JSON 数组")
    gold_tune, gold_regress = split_gold_tune_regress(gold, tune_size=tune_size, seed=seed)
    gold_tune_path = ROOT / "data" / "gold" / "gold_tune.json"
    gold_regress_path = ROOT / "data" / "gold" / "gold_regress.json"
    for path in (gold_tune_path, gold_regress_path):
        if path.exists():
            raise FileExistsError(f"目标金标拆分已存在，拒绝覆盖: {path}")
    gold_tune_path.write_text(json.dumps(gold_tune, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gold_regress_path.write_text(json.dumps(gold_regress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    leak = leakage_report({"gold_tune": gold_tune, "gold_regress": gold_regress})
    if any(leak["exact_overlaps"].values()) or any(leak["template_overlaps_report_only"].values()):
        raise RuntimeError(f"旧金标拆分存在泄漏: {leak}")
    outputs = {}
    for name, path, values in (
        ("gold_tune", gold_tune_path, gold_tune),
        ("gold_regress", gold_regress_path, gold_regress),
    ):
        outputs[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "count": len(values),
            "labels": dict(Counter(str(item.get("gold_verdict", "")) for item in values)),
        }
    manifest = {
        "version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "gold_source": {"path": str(gold_path.resolve()), "sha256": sha256_file(gold_path), "count": len(gold)},
        "policy": {
            "gold_split": "semantic-stratified template-group split; 220 tune / 100 regress",
            "training_use": "none; both partitions are excluded from model fitting",
            "gold_tune_use": "decision thresholds and rules only",
            "gold_regress_use": "v1/v2 regression comparison only",
        },
        "outputs": outputs,
        "leakage": leak,
    }
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFESTS / "gold_split_v2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_final_window_v2(source_path: Path, tag: str = "v2") -> dict[str, Any]:
    """清洗全新窗口并与全部历史分区做精确+模板双重隔离。"""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", tag):
        raise ValueError("tag 只能包含字母、数字、下划线和连字符")
    output = PROCESSED / f"final_test_{tag}.jsonl"
    manifest_path = MANIFESTS / f"final_window_{tag}_manifest.json"
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"目标窗口已存在，拒绝覆盖: {output}")
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() == ".jsonl":
        incoming = list(_read_jsonl(source_path))
    else:
        value = json.loads(source_path.read_text(encoding="utf-8"))
        incoming = value if isinstance(value, list) else [value]
    normalized = [standardize(item) for item in incoming]
    invalid_labels = Counter(item.get("label", "") for item in normalized if item.get("label") not in {"confirmed_upload_success", "failed"})
    if invalid_labels:
        raise ValueError(f"新窗口含非 TDP 二分类标签: {dict(invalid_labels)}")

    history_paths = {
        "train": PROCESSED / "train.jsonl",
        "val": PROCESSED / "val.jsonl",
        "old_final_test": PROCESSED / "final_test.jsonl",
        "old_gold": GOLD_V1,
    }
    history_rows = {}
    for name, path in history_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"历史分区不存在: {path}")
        history_rows[name] = (
            list(_read_jsonl(path))
            if path.suffix.lower() == ".jsonl"
            else json.loads(path.read_text(encoding="utf-8"))
        )

    blocked_exact = {exact_fingerprint(item) for rows in history_rows.values() for item in rows}
    blocked_template = {template_fingerprint(item) for rows in history_rows.values() for item in rows}
    after_history = [
        item for item in normalized
        if exact_fingerprint(item) not in blocked_exact and template_fingerprint(item) not in blocked_template
    ]
    final_rows, dedupe_stats, conflicts = _dedupe(after_history, template_cap=1, exclude_template_conflicts=True)
    if len(final_rows) < 3300:
        raise RuntimeError(f"去重后仅 {len(final_rows)} 条；抽取约 300 条金标后无法保证 final_test_{tag} >= 3000")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, final_rows)
    overlap = leakage_report({**history_rows, f"final_test_{tag}": final_rows})
    final_keys = [key for key in overlap["exact_overlaps"] if f"final_test_{tag}" in key]
    final_template_keys = [key for key in overlap["template_overlaps_report_only"] if f"final_test_{tag}" in key]
    if any(overlap["exact_overlaps"][key] for key in final_keys) or any(
        overlap["template_overlaps_report_only"][key] for key in final_template_keys
    ):
        raise RuntimeError("新窗口与历史分区仍存在指纹重叠")
    manifest = {
        "version": 2,
        "tag": tag,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source_path.resolve()), "sha256": sha256_file(source_path), "count": len(incoming)},
        "policy": "history exact+template exclusion; internal exact+template dedupe; TDP binary labels only",
        "after_history_exclusion": len(after_history),
        "dedupe": dedupe_stats,
        "conflicts_excluded": len(conflicts),
        "output": {
            "path": str(output.resolve()),
            "sha256": sha256_file(output),
            "count_before_gold_deduction": len(final_rows),
            "labels": dict(Counter(item["label"] for item in final_rows)),
        },
        "history": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in history_paths.items()
        },
        "leakage": overlap,
        "gold_deduction": {"status": "pending"},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(MANIFESTS / f"final_window_{tag}_conflicts.jsonl", conflicts)
    return manifest


def _dedupe(
    rows: list[dict[str, Any]],
    template_cap: int,
    exclude_template_conflicts: bool,
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    exact_labels: dict[str, set[str]] = defaultdict(set)
    for item in rows:
        exact_labels[exact_fingerprint(item)].add(item["label"])
    exact_conflicts = {fingerprint for fingerprint, labels in exact_labels.items() if len(labels) > 1}
    conflicts = [
        {"fingerprint_type": "exact", "fingerprint": fingerprint, "labels": sorted(exact_labels[fingerprint]), "resolution": "unknown_excluded"}
        for fingerprint in sorted(exact_conflicts)
    ]
    seen_exact: set[str] = set()
    exact_unique: list[dict[str, Any]] = []
    for item in rows:
        fingerprint = exact_fingerprint(item)
        if fingerprint in exact_conflicts or fingerprint in seen_exact:
            continue
        seen_exact.add(fingerprint)
        exact_unique.append(item)
    if template_cap < 1:
        return exact_unique, {"input": len(rows), "after_exact": len(exact_unique), "after_template": len(exact_unique)}, conflicts

    template_labels: dict[str, set[str]] = defaultdict(set)
    for item in exact_unique:
        template_labels[template_fingerprint(item)].add(item["label"])
    template_conflicts = {fingerprint for fingerprint, labels in template_labels.items() if len(labels) > 1}
    conflicts.extend(
        {"fingerprint_type": "template", "fingerprint": fingerprint, "labels": sorted(template_labels[fingerprint]), "resolution": "unknown_excluded"}
        for fingerprint in sorted(template_conflicts)
    )
    seen_template: Counter[str] = Counter()
    unique: list[dict[str, Any]] = []
    for item in exact_unique:
        fingerprint = template_fingerprint(item)
        dedupe_key = fingerprint if exclude_template_conflicts else f"{fingerprint}:{item['label']}"
        if (exclude_template_conflicts and fingerprint in template_conflicts) or seen_template[dedupe_key] >= template_cap:
            continue
        seen_template[dedupe_key] += 1
        unique.append(item)
    return unique, {
        "input": len(rows),
        "exact_conflicts": len(exact_conflicts),
        "after_exact": len(exact_unique),
        "template_conflicts": len(template_conflicts),
        "template_conflicts_excluded": exclude_template_conflicts,
        "template_cap": template_cap,
        "after_template": len(unique),
    }, conflicts


def _stratified_split(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        grouped[item["label"]].append(item)
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(grouped):
        values = grouped[label]
        rng.shuffle(values)
        val_size = max(1, round(len(values) * val_ratio))
        val.extend(values[:val_size])
        train.extend(values[val_size:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _normalize_label(label: str) -> str:
    return "confirmed_upload_success" if label in {"success", "likely_upload_success", "confirmed_upload_success"} else "failed" if label in {"failed", "failure"} else label


def _parse_header_text(text: str) -> dict[str, str]:
    result = {}
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if ":" in line:
            name, value = line.split(":", 1)
            if name.strip():
                result[name.strip()] = value.strip()
    return result


def _response_format(item: dict[str, Any]) -> str:
    body = item["response"].get("body", "").strip()
    content_type = item["response"].get("content_type", "").lower()
    if not body:
        return "empty"
    if "json" in content_type or body[:1] in "{[":
        return "json"
    if "html" in content_type or re.search(r"<(?:html|body|form)\b", body, re.I):
        return "html"
    return "text"


def _status_family(item: dict[str, Any]) -> str:
    code = int(item["response"].get("status_code", 0))
    return f"{code // 100}xx" if code else "unknown"


def _extension_group(item: dict[str, Any]) -> str:
    ext = str(item["request"].get("file_ext", "")).lower().strip(".")
    if ext in {"php", "jsp", "aspx"}:
        return "php_jsp_aspx"
    if ext in {"step", "stp", "stl"}:
        return "step_stl"
    return "other"


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--without-raw", action="store_true", help="只在诊断时禁用可选 raw 补充")
    parser.add_argument("--report-leak", action="store_true")
    parser.add_argument("--split-gold", action="store_true", help="把旧金标拆为 gold_tune/gold_regress；均不进训练")
    parser.add_argument("--gold-path", type=Path, default=GOLD_V1)
    parser.add_argument("--gold-tune-size", type=int, default=220)
    parser.add_argument("--final-window", type=Path)
    parser.add_argument("--tag", default="v2")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.split_gold:
        manifest = split_gold_for_v2(args.gold_path, tune_size=args.gold_tune_size, seed=args.seed)
        print(json.dumps({"outputs": manifest["outputs"], "leakage": manifest["leakage"]}, ensure_ascii=False, indent=2))
        return 0
    if args.final_window:
        manifest = build_final_window_v2(args.final_window, args.tag)
        print(json.dumps({"output": manifest["output"], "gold_deduction": manifest["gold_deduction"]}, ensure_ascii=False, indent=2))
        return 0
    if args.report_leak:
        report_existing()
        return 0
    manifest = build(args.source_root, include_raw=not args.without_raw, seed=args.seed)
    print(json.dumps({"outputs": manifest["outputs"], "leakage": manifest["leakage"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
