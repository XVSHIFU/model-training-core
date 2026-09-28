"""构建 v4 训练集：train_v3 + 两批 behavior Codex 金标。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = (
    ROOT / "data" / "test"
    if (ROOT / "data" / "test" / "20260812-123235-96503a").exists()
    else ROOT / "data" / "test-v3"
)
DEFAULT_BASE = ROOT / "data" / "processed" / "train_v3.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "train_v4.jsonl"
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "train_v4_manifest.json"
DEFAULT_BEHAVIOR_SOURCES = (
    (
        TEST_ROOT / "20260812-123235-96503a" / "upload_behavior_500.jsonl",
        TEST_ROOT / "20260812-123235-96503a" / "codex_behavior_500.jsonl",
    ),
    (
        TEST_ROOT / "20260812-133047-328e70" / "upload_behavior_500.jsonl",
        TEST_ROOT / "20260812-133047-328e70" / "codex_behavior_500.jsonl",
    ),
)
VERDICTS = {"confirmed_upload_success", "likely_upload_success", "failed", "unknown"}
SUCCESS_VERDICTS = {"confirmed_upload_success", "likely_upload_success"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def core_training_row(row: dict[str, Any], label: str, source: str) -> dict[str, Any]:
    if label not in VERDICTS:
        raise ValueError(f"非法四分类标签: {label}")
    return {
        "event_id": str(row["event_id"]),
        "request": dict(row["request"]),
        "response": dict(row["response"]),
        "label": label,
        "v4_source": source,
    }


def load_behavior_gold(events_path: Path, labels_path: Path) -> list[dict[str, Any]]:
    events = {str(row["event_id"]): row for row in read_jsonl(events_path)}
    labels: dict[str, str] = {}
    for item in read_jsonl(labels_path):
        event_id = str(item["event_id"])
        verdict = str(item.get("codex_verdict", item.get("gold_verdict", "")))
        if verdict not in VERDICTS:
            raise ValueError(f"{labels_path} 含非法标签: {event_id}={verdict}")
        previous = labels.get(event_id)
        if previous is not None and previous != verdict:
            raise ValueError(f"Codex 金标冲突: {event_id}: {previous} vs {verdict}")
        labels[event_id] = verdict
    missing_events = set(labels) - set(events)
    missing_labels = set(events) - set(labels)
    if missing_events or missing_labels:
        raise ValueError(
            f"行为事件/金标 event_id 不完全对齐: missing_events={len(missing_events)}, "
            f"missing_labels={len(missing_labels)}"
        )
    return [core_training_row(events[event_id], labels[event_id], "behavior_codex_gold") for event_id in sorted(labels)]


def build_train_v4(
    base_path: Path = DEFAULT_BASE,
    behavior_sources: tuple[tuple[Path, Path], ...] = DEFAULT_BEHAVIOR_SOURCES,
    output_path: Path = DEFAULT_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    base_rows = read_jsonl(base_path)
    chosen: dict[str, tuple[int, dict[str, Any]]] = {}
    for row in base_rows:
        label = str(row.get("label", ""))
        item = core_training_row(row, label, str(row.get("v3_source", "train_v3")))
        chosen[item["event_id"]] = (10, item)

    gold_rows: list[dict[str, Any]] = []
    input_manifest: list[dict[str, Any]] = []
    for events_path, labels_path in behavior_sources:
        batch = load_behavior_gold(events_path, labels_path)
        gold_rows.extend(batch)
        input_manifest.append(
            {
                "events": {"path": str(events_path.resolve()), "sha256": sha256_file(events_path), "count": len(batch)},
                "labels": {"path": str(labels_path.resolve()), "sha256": sha256_file(labels_path), "count": len(batch)},
            }
        )

    gold_by_id: dict[str, dict[str, Any]] = {}
    for row in gold_rows:
        event_id = str(row["event_id"])
        previous = gold_by_id.get(event_id)
        if previous is not None and previous["label"] != row["label"]:
            raise ValueError(f"两批 Codex 金标冲突: {event_id}")
        gold_by_id[event_id] = row
    overlaps = set(chosen) & set(gold_by_id)
    for event_id, row in gold_by_id.items():
        chosen[event_id] = (20, row)

    output = [chosen[event_id][1] for event_id in sorted(chosen)]
    if len(output) != len({str(row["event_id"]) for row in output}):
        raise RuntimeError("train_v4 event_id 不唯一")
    write_jsonl(output_path, output)

    counts = Counter(str(row["label"]) for row in output)
    source_counts = Counter(str(row["v4_source"]) for row in output)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "pipelines/build_train_v4.py",
        "label_priority": ["behavior_codex_gold", "train_v3"],
        "binary_mapping": {
            "success": sorted(SUCCESS_VERDICTS),
            "failed": ["failed"],
            "excluded": ["unknown"],
        },
        "evaluation_note": "两批 behavior Codex 金标同时用于计划指定的回流与开发验收，不是新盲测。",
        "inputs": {
            "train_v3": {"path": str(base_path.resolve()), "sha256": sha256_file(base_path), "count": len(base_rows)},
            "behavior_codex_batches": input_manifest,
        },
        "codex_gold_count": len(gold_by_id),
        "codex_override_count": len(overlaps),
        "output": {"path": str(output_path.resolve()), "sha256": sha256_file(output_path), "count": len(output)},
        "label_counts": dict(counts),
        "source_counts": dict(source_counts),
        "binary_training_counts": {
            "success": sum(counts[label] for label in SUCCESS_VERDICTS),
            "failed": counts["failed"],
            "excluded_unknown": counts["unknown"],
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_train_v4(args.base, DEFAULT_BEHAVIOR_SOURCES, args.output, args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
