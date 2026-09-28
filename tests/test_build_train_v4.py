import json

import pytest

from pipelines.build_train_v4 import build_train_v4


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _event(event_id, label="unknown"):
    return {
        "event_id": event_id,
        "request": {"filename": "a.txt"},
        "response": {"status_code": 200, "body": "{}"},
        "label": label,
    }


def test_build_train_v4_codex_gold_overrides_base_and_preserves_unknown(tmp_path):
    base = tmp_path / "base.jsonl"
    events = tmp_path / "events.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "train_v4.jsonl"
    manifest = tmp_path / "manifest.json"
    _write_jsonl(base, [_event("same", "failed"), _event("base", "unknown")])
    _write_jsonl(events, [_event("same"), _event("new")])
    _write_jsonl(
        labels,
        [
            {"event_id": "same", "codex_verdict": "confirmed_upload_success"},
            {"event_id": "new", "codex_verdict": "unknown"},
        ],
    )

    result = build_train_v4(base, ((events, labels),), output, manifest)
    rows = {row["event_id"]: row for row in map(json.loads, output.read_text(encoding="utf-8").splitlines())}
    assert rows["same"]["label"] == "confirmed_upload_success"
    assert rows["same"]["v4_source"] == "behavior_codex_gold"
    assert rows["base"]["label"] == "unknown"
    assert result["codex_override_count"] == 1
    assert result["binary_training_counts"]["excluded_unknown"] == 2


def test_build_train_v4_rejects_unaligned_gold(tmp_path):
    base = tmp_path / "base.jsonl"
    events = tmp_path / "events.jsonl"
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(base, [_event("base", "failed")])
    _write_jsonl(events, [_event("event")])
    _write_jsonl(labels, [{"event_id": "other", "codex_verdict": "failed"}])
    with pytest.raises(ValueError, match="不完全对齐"):
        build_train_v4(base, ((events, labels),), tmp_path / "out.jsonl", tmp_path / "manifest.json")
