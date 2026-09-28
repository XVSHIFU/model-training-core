"""Deterministic, task-independent data loading and supervision handling."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel

from training_core.contracts import Sample, TaskAdapter


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"Unsupported value for stable hashing: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=_json_value)


def stable_hash(value: Any) -> str:
    """Hash JSON data independently of dictionary insertion order."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON object/array or JSONL, reporting locations without raw data."""
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as stream:
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{line_number}")
                records.append(record)
        else:
            try:
                value = json.load(stream)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {exc.lineno}") from exc
            records = [value] if isinstance(value, dict) else value
            if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
                raise ValueError(f"Expected a JSON object or an array of objects in {path}")
    return records


def _validate_samples(samples: Iterable[Sample], context: str) -> None:
    seen: set[str] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, Sample):
            raise ValueError(f"Expected Sample at {context}, position {index}")
        if not isinstance(sample.sample_id, str) or not sample.sample_id.strip():
            raise ValueError(f"Sample ID must be a nonempty string at {context}, position {index}")
        if sample.sample_id in seen:
            raise ValueError(f"Duplicate sample ID in {context}: {sample.sample_id}")
        seen.add(sample.sample_id)


def load_samples(path: str | Path, adapter: TaskAdapter) -> list[Sample]:
    samples = [adapter.parse(record) for record in read_records(path)]
    _validate_samples(samples, str(path))
    return samples


def prepare_targets(
    samples: list[Sample],
    mapping: dict,
    excluded: set[str],
) -> tuple[list[Sample], dict[str, Any]]:
    """Select supervised rows without changing their original targets or metadata.

    ``mapping`` and ``excluded`` define the task's legal original labels. Label
    status controls availability; it is never a prediction's abstention status.
    """
    if set(mapping) & set(excluded):
        raise ValueError("A target cannot be both mapped and explicitly excluded")
    _validate_samples(samples, "target preparation")
    kept: list[Sample] = []
    reasons: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    for sample in samples:
        if sample.label_status not in {"labeled", "unlabeled", "unresolved", "conflict"}:
            raise ValueError(f"Unrecognized label status for sample {sample.sample_id}")
        target = sample.target
        # Validate labels even on unresolved rows, so typos cannot disappear.
        if target is not None:
            try:
                legal = target in mapping or target in excluded
            except TypeError as exc:
                raise ValueError(f"Unhashable target for sample {sample.sample_id}") from exc
            if not legal:
                raise ValueError(f"Unrecognized target for sample {sample.sample_id}")
        if sample.label_status in {"unresolved", "conflict"}:
            reasons[f"status:{sample.label_status}"] += 1
        elif target is None or sample.label_status == "unlabeled":
            reasons["unlabeled"] += 1
        elif target in excluded:
            reasons[f"target:{target}"] += 1
        else:
            training_target = mapping[target]
            kept.append(replace(sample, metadata={**sample.metadata, "training_target": training_target}))
            class_counts[str(training_target)] += 1
    return kept, {
        "total": len(samples),
        "used": len(kept),
        "excluded": len(samples) - len(kept),
        "reasons": dict(sorted(reasons.items())),
        "class_counts": dict(sorted(class_counts.items())),
    }


def merge_samples(sources: list[tuple[int, list[Sample]]]) -> tuple[list[Sample], dict[str, Any]]:
    """Choose the highest-priority source per ID and retain a deterministic audit.

    Equal-priority disagreement in inputs, targets, group or label status is an
    error. Equivalent annotations with different provenance choose a stable
    representative, with every source retained in the audit.
    """
    grouped: dict[str, list[tuple[int, Sample]]] = defaultdict(list)
    for priority, samples in sources:
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise ValueError("Source priority must be an integer")
        _validate_samples(samples, f"source with priority {priority}")
        for sample in samples:
            grouped[sample.sample_id].append((priority, sample))
    output: list[Sample] = []
    decisions: list[dict[str, Any]] = []
    conflict_count = 0
    for sample_id, candidates in sorted(grouped.items()):
        by_priority: dict[int, set[str]] = defaultdict(set)
        for priority, sample in candidates:
            by_priority[priority].add(stable_hash({
                "inputs": sample.inputs, "target": sample.target,
                "group_id": sample.group_id, "label_status": sample.label_status,
            }))
        if any(len(signatures) > 1 for signatures in by_priority.values()):
            raise ValueError(f"Conflicting samples at equal priority: {sample_id}")
        ordered = sorted(candidates, key=lambda entry: (-entry[0], _canonical_json(asdict(entry[1]))))
        priority, selected = ordered[0]
        output.append(replace(selected, metadata=dict(selected.metadata)))
        if len(ordered) > 1:
            target_conflict = len({stable_hash(sample.target) for _, sample in ordered}) > 1
            conflict_count += int(target_conflict)
            decisions.append({
                "sample_id": sample_id,
                "selected_priority": priority,
                "selected_source": selected.source,
                "target_conflict": target_conflict,
                "candidates": [
                    {"priority": level, "source": sample.source, "target": sample.target,
                     "label_status": sample.label_status, "input_hash": stable_hash(sample.inputs)}
                    for level, sample in ordered
                ],
            })
    count = sum(len(samples) for _, samples in sources)
    return output, {
        "input": count, "output": len(output), "duplicates": count - len(output),
        "conflicts": conflict_count, "decisions": decisions,
    }
