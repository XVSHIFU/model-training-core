"""Boundary checks for supervision, deterministic merging and dataset purpose."""
from dataclasses import asdict
import json

import pytest

from training_core.contracts import DatasetSpec, Sample
from training_core.data import load_samples, merge_samples, prepare_targets, read_records, sha256_file, stable_hash
from training_core.splits import build_lineage, check_lineage, inspect_splits, validate_lineage_report, validate_splits


class Adapter:
    def parse(self, row):
        return Sample(sample_id=row["id"], inputs=row.get("inputs", {}), target=row.get("target"))


def sample(identifier, target="yes", **kwargs):
    return Sample(identifier, kwargs.pop("inputs", {"text": identifier}), target=target,
                  label_status=kwargs.pop("label_status", "labeled"), **kwargs)


def spec(purpose, **kwargs):
    return DatasetSpec(path="unused.jsonl", purpose=purpose, **kwargs)


def fingerprints(row):
    return row.metadata.get("fingerprints", {})


@pytest.mark.parametrize("extension", ["json", "jsonl"])
def test_loading_preserves_order_and_checks_ids(tmp_path, extension):
    path = tmp_path / f"data.{extension}"
    rows = [{"id": "b", "target": "yes"}, {"id": "a", "target": "no"}]
    path.write_text(json.dumps(rows) if extension == "json" else "\n".join(map(json.dumps, rows)), encoding="utf-8-sig")
    assert [row.sample_id for row in load_samples(path, Adapter())] == ["b", "a"]
    assert read_records(path) == rows
    assert len(sha256_file(path)) == 64
    for invalid in ([{"id": " "}], [{"id": "a"}, {"id": "a"}]):
        path.write_text(json.dumps(invalid) if extension == "json" else "\n".join(map(json.dumps, invalid)), encoding="utf-8")
        with pytest.raises(ValueError):
            load_samples(path, Adapter())


@pytest.mark.parametrize("contents", ["null", "[1]", '"text"', "{invalid"])
def test_records_reject_nonobject_or_invalid_json(tmp_path, contents):
    path = tmp_path / "input.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        read_records(path)


def test_prepare_targets_keeps_original_targets_and_separates_exclusions():
    rows = [sample("a", metadata={"keep": True}), sample("b", "no"), sample("c", None),
            sample("d", "unknown"), sample("e", "yes", label_status="unresolved"),
            sample("f", "no", label_status="conflict")]
    before = [asdict(row) for row in rows]
    kept, report = prepare_targets(rows, {"yes": 1, "no": 0}, {"unknown"})
    assert [row.target for row in kept] == ["yes", "no"]
    assert [row.metadata["training_target"] for row in kept] == [1, 0]
    assert kept[0].metadata["keep"] is True
    assert [asdict(row) for row in rows] == before
    assert report == {"total": 6, "used": 2, "excluded": 4,
                      "reasons": {"status:conflict": 1, "status:unresolved": 1, "target:unknown": 1, "unlabeled": 1},
                      "class_counts": {"0": 1, "1": 1}}
    with pytest.raises(ValueError, match="Unrecognized"):
        prepare_targets([sample("typo", "yess", label_status="unresolved")], {"yes": 1}, set())
    with pytest.raises(ValueError, match="both"):
        prepare_targets([], {"yes": 1}, {"yes"})
    with pytest.raises(ValueError, match="label status"):
        prepare_targets([sample("x", "yes", label_status="lablled")], {"yes": 1}, set())
    assert prepare_targets([sample("x", "yes", label_status="unlabeled")], {"yes": 1}, set())[1]["reasons"] == {"unlabeled": 1}


def test_merge_is_deterministic_retains_conflicts_and_rejects_ties():
    lower = [sample("a", "no", source="platform"), sample("b", "yes")]
    higher = [sample("a", "yes", source="reviewed")]
    merged, audit = merge_samples([(10, lower), (20, higher)])
    reverse, reversed_audit = merge_samples([(20, higher), (10, list(reversed(lower)))])
    assert [asdict(row) for row in merged] == [asdict(row) for row in reverse]
    assert audit == reversed_audit
    assert [row.target for row in merged] == ["yes", "yes"]
    assert audit["conflicts"] == 1
    assert {entry["target"] for entry in audit["decisions"][0]["candidates"]} == {"yes", "no"}
    assert lower[0].target == "no"
    with pytest.raises(ValueError, match="equal priority"):
        merge_samples([(10, lower), (10, higher)])
    with pytest.raises(ValueError, match="equal priority"):
        merge_samples([(10, [sample("a")]), (10, [sample("a", inputs={"text": "different"})])])
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


@pytest.mark.parametrize("overlap_kind", ["id", "exact", "template", "group"])
def test_overlap_detected_independent_rejected_history_allowed(overlap_kind):
    left, right = sample("a"), sample("b")
    if overlap_kind == "id":
        right.sample_id = "a"
    elif overlap_kind == "group":
        left.group_id = right.group_id = "shared"
    else:
        left.metadata = right.metadata = {"fingerprints": {overlap_kind: "shared"}}
    data = {"training": [left], "evaluation": [right]}
    specs = {"training": spec("train"), "evaluation": spec("independent_test", independent=True, provenance="frozen source")}
    report = inspect_splits(data, specs, fingerprints)
    assert report["pairs"][0]["overlaps"][overlap_kind] == 1
    assert not report["independent_test_eligible"]
    with pytest.raises(ValueError, match="independent"):
        validate_splits(report, require_independent=True)
    validate_splits(report, for_training=True)  # Unused test does not prevent training.
    specs["evaluation"] = spec("historical_regression")
    history = inspect_splits(data, specs, fingerprints)
    assert history["valid"]
    assert history["pairs"] == [{**report["pairs"][0], "left_purpose": "historical_regression"}]
    validate_splits(history)
    specs["evaluation"] = spec("development")
    with pytest.raises(ValueError, match="Training/development"):
        validate_splits(inspect_splits(data, specs, fingerprints), for_training=True)


def test_empty_fingerprints_skipped_and_independence_requires_evidence():
    data = {"train": [sample("a")], "test": [sample("b", metadata={"fingerprints": {"exact": "", "template": " "}})]}
    specs = {"train": spec("train"), "test": spec("independent_test", independent=True, provenance="new window")}
    report = inspect_splits(data, specs, fingerprints)
    assert report["pairs"][0]["overlaps"] == {"id": 0, "exact": 0, "template": 0, "group": 0}
    validate_splits(report, require_independent=True)
    assert report == inspect_splits(dict(reversed(list(data.items()))), dict(reversed(list(specs.items()))), fingerprints)
    for details in ({"independent": False, "provenance": "source"}, {"independent": True, "provenance": " "}):
        specs["test"] = spec("independent_test", **details)
        with pytest.raises(ValueError, match="independent"):
            validate_splits(inspect_splits(data, specs, fingerprints), require_independent=True)
    with pytest.raises(ValueError, match="independent"):
        validate_splits(inspect_splits({"train": data["train"]}, {"train": specs["train"]}, fingerprints), require_independent=True)


def test_independent_test_also_excludes_history_and_empty_data():
    data = {"history": [sample("a")], "test": [sample("a")]}
    specs = {"history": spec("historical_regression"), "test": spec("independent_test", independent=True, provenance="attested")}
    assert not inspect_splits(data, specs, fingerprints)["independent_test_eligible"]
    data["test"] = []
    assert not inspect_splits(data, specs, fingerprints)["independent_test_eligible"]


def test_stable_hash_and_merge_support_validated_task_objects():
    from pydantic import BaseModel

    class TaskInput(BaseModel):
        text: str

    inputs = TaskInput(text="task data")
    assert stable_hash(inputs) == stable_hash(inputs.model_dump(mode="json"))
    rows, report = merge_samples([(1, [sample("a", inputs=inputs)]), (2, [sample("a", inputs=inputs)])])
    assert isinstance(rows[0].inputs, TaskInput)
    assert report["duplicates"] == 1


@pytest.mark.parametrize("kind", ["id", "exact", "template", "group"])
def test_portable_lineage_blocks_overlap_without_original_training_files(tmp_path, kind):
    training = sample("private-training-id", inputs={"secret": "do not persist"}, group_id="private-group")
    training.metadata["fingerprints"] = {"exact": "private-content-key", "template": "private-template-key"}
    train_path = tmp_path / "original.jsonl"
    train_path.write_text("private original data", encoding="utf-8")
    lineage = build_lineage({"fit": [training]}, {"fit": DatasetSpec(path=str(train_path), purpose="train")}, fingerprints)
    serialized = json.dumps(lineage)
    assert not any(value in serialized for value in ["private-training-id", "private-group", "private-content-key", "private-template-key", "secret", "do not persist", str(train_path)])
    moved = tmp_path / "moved-artifact"
    moved.mkdir()
    (moved / "lineage.json").write_text(serialized, encoding="utf-8")
    train_path.unlink()  # Simulate an artifact moved to a machine without source files.
    restored = json.loads((moved / "lineage.json").read_text(encoding="utf-8"))
    incoming = sample("new-id")
    if kind == "id":
        incoming.sample_id = training.sample_id
    elif kind == "group":
        incoming.group_id = training.group_id
    else:
        incoming.metadata["fingerprints"] = {kind: training.metadata["fingerprints"][kind]}
    test_spec = spec("independent_test", independent=True, provenance="new frozen window")
    report = check_lineage([incoming], test_spec, fingerprints, restored)
    assert report["overlaps"][kind] == 1
    assert report["entries"][0]["overlaps"][kind] == 1
    assert not report["independent_test_eligible"]
    with pytest.raises(ValueError, match="lineage_overlap"):
        validate_lineage_report(report)
    history = check_lineage([incoming], spec("historical_regression"), fingerprints, restored)
    assert history["eligible"] and not history["independent_test_eligible"]
    assert history["overlaps"] == report["overlaps"]
    validate_lineage_report(history)


def test_lineage_requires_training_membership_and_evidence_for_independent_test():
    training, incoming = sample("train"), sample("fresh")
    lineage = build_lineage({"fit": [training]}, {"fit": spec("train")}, fingerprints)
    independent = spec("independent_test", independent=True, provenance="documented source")
    validate_lineage_report(check_lineage([incoming], independent, fingerprints, lineage))
    for absent in [None, {"version": 1, "entries": {}}, build_lineage({"dev": [training]}, {"dev": spec("development")}, fingerprints)]:
        report = check_lineage([incoming], independent, fingerprints, absent)
        assert "missing_training_lineage" in report["violations"]
        with pytest.raises(ValueError, match="missing_training_lineage"):
            validate_lineage_report(report)
    for invalid_spec in [spec("independent_test", provenance="source"), spec("independent_test", independent=True)]:
        assert not check_lineage([incoming], invalid_spec, fingerprints, lineage)["eligible"]
    assert not check_lineage([], independent, fingerprints, lineage)["eligible"]
    history = check_lineage([incoming], spec("historical_regression"), fingerprints, None)
    assert history["eligible"] and not history["lineage_available"]
    lineage["entries"]["fit"]["keys"]["id"] = []
    with pytest.raises(ValueError, match="IDs/count"):
        check_lineage([incoming], independent, fingerprints, lineage)


def test_lineage_records_development_and_counts_union_without_duplicates():
    training, development = sample("a"), sample("b")
    shared = {"fingerprints": {"template": "same"}}
    training.metadata = development.metadata = shared
    datasets = {"fit": [training], "selection": [development]}
    specs = {"fit": spec("train"), "selection": spec("development")}
    lineage = build_lineage(datasets, specs, fingerprints)
    assert lineage == build_lineage(dict(reversed(list(datasets.items()))), dict(reversed(list(specs.items()))), fingerprints)
    incoming = sample("c", metadata=shared)
    report = check_lineage([incoming], spec("independent_test", independent=True, provenance="source"), fingerprints, lineage)
    assert report["overlaps"]["template"] == 1
    assert sum(entry["overlaps"]["template"] for entry in report["entries"]) == 2
