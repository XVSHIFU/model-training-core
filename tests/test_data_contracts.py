"""Synthetic failures at the source-bytes, identity and input-schema boundaries."""
import hashlib
import json
from pathlib import Path

import pytest

from training_core.contracts import Sample
from training_core.data import load_samples, load_samples_snapshot, prepare_targets, read_records, read_records_snapshot, sha256_file
from training_tasks.text_classification import TextClassificationTask


def record(**changes):
    return {"sample_id": "synthetic-id", "inputs": {"text": "synthetic topic text", "features": []},
            "target": "topic", **changes}


@pytest.mark.parametrize("identifier", [None, 0, 17, False, [], "", " \t"])
def test_text_ids_are_rejected_instead_of_stringified(identifier):
    with pytest.raises(ValueError, match="ID"):
        TextClassificationTask().parse(record(sample_id=identifier))


@pytest.mark.parametrize("field,value", [
    ("group_id", 17), ("group_id", ["group"]), ("group_id", {}), ("group_id", " "),
    ("source", None), ("source", 17), ("metadata", None), ("metadata", [["key", "value"]]),
    ("label_status", None), ("label_status", ["labeled"]), ("label_status", "labelled"),
])
def test_metadata_contract_does_not_silently_coerce(field, value):
    with pytest.raises(ValueError):
        TextClassificationTask().parse(record(**{field: value}))


@pytest.mark.parametrize("feature", [float("nan"), float("inf"), -float("inf"), "NaN", "1.2", True])
def test_text_features_must_be_finite_numeric_values(feature):
    with pytest.raises(ValueError):
        TextClassificationTask().parse(record(inputs={"text": "valid text", "features": [feature]}))


def test_valid_ids_and_numeric_features_are_preserved_and_blank_text_is_rejected():
    row = TextClassificationTask().parse(record(sample_id="记录-零", inputs={"text": " valid ", "features": [0, 2, -1.5]}))
    assert row.sample_id == "记录-零" and row.inputs["features"] == [0.0, 2.0, -1.5]
    assert row.inputs["text"] == " valid "
    with pytest.raises(ValueError, match="blank"):
        TextClassificationTask().parse(record(inputs={"text": " \t\n", "features": []}))
    with pytest.raises(ValueError, match="ID"):
        TextClassificationTask().parse({"inputs": {"text": "valid"}})


def test_mutated_sample_is_revalidated_before_supervised_use():
    row = Sample("id", {}, "yes", label_status="labeled")
    row.label_status = ["labeled"]
    with pytest.raises(ValueError, match="label status"):
        prepare_targets([row], {"yes": 1}, set())
    row.label_status = "labeled"
    row.group_id = 17
    with pytest.raises(ValueError, match="group_id"):
        prepare_targets([row], {"yes": 1}, set())


@pytest.mark.parametrize("extension", ["json", "jsonl"])
def test_snapshot_digest_identifies_loaded_bytes_even_when_parse_replaces_source(tmp_path, monkeypatch, extension):
    path = tmp_path / f"samples.{extension}"
    rows = [record(sample_id="original-a"), record(sample_id="original-b")]
    text = json.dumps(rows) if extension == "json" else "\r\n".join(json.dumps(row) for row in rows) + "\r\n"
    original = b"\xef\xbb\xbf" + text.encode("utf-8")
    replacement = json.dumps(record(sample_id="replacement")).encode("utf-8")
    path.write_bytes(original)
    real_reader = Path.read_bytes
    calls = []

    def read_once(candidate):
        if candidate == path:
            calls.append(candidate)
        return real_reader(candidate)

    class ReplacingAdapter(TextClassificationTask):
        def parse(self, item):
            path.write_bytes(replacement)
            return super().parse(item)

    monkeypatch.setattr(Path, "read_bytes", read_once)
    loaded, digest = load_samples_snapshot(path, ReplacingAdapter())
    assert [sample.sample_id for sample in loaded] == ["original-a", "original-b"]
    assert digest == hashlib.sha256(original).hexdigest()
    assert digest != sha256_file(path)
    assert len(calls) == 1
    assert load_samples(path, TextClassificationTask())[0].sample_id == "replacement"


@pytest.mark.parametrize("extension", ["json", "jsonl"])
@pytest.mark.parametrize("invalid_number", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_json_is_rejected_before_task_parsing(tmp_path, extension, invalid_number):
    path = tmp_path / f"records.{extension}"
    path.write_text('{"sample_id":"id","value":' + invalid_number + '}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        read_records(path)


def test_record_snapshot_hashes_raw_encoding_not_reserialized_records(tmp_path):
    path = tmp_path / "rows.jsonl"
    raw = b'\xef\xbb\xbf{ "id": "a" }\r\n\r\n{"id":"b"}\r\n'
    path.write_bytes(raw)
    records, digest = read_records_snapshot(path)
    assert records == [{"id": "a"}, {"id": "b"}]
    assert digest == hashlib.sha256(raw).hexdigest()


def test_jsonl_unicode_separators_inside_text_are_not_record_boundaries(tmp_path):
    path = tmp_path / "unicode.jsonl"
    rows = [record(sample_id="unicode", inputs={"text": "first\u2028second\u2029third", "features": []}), record(sample_id="next")]
    path.write_bytes("\r\n".join(json.dumps(row, ensure_ascii=False) for row in rows).encode("utf-8"))
    assert read_records(path) == rows
    samples, _ = load_samples_snapshot(path, TextClassificationTask())
    assert samples[0].inputs["text"] == "first\u2028second\u2029third"
