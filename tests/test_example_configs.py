"""Run public synthetic examples through real adapters and portable lineage."""
from collections import Counter
import json
from pathlib import Path
import runpy

from training_core.contracts import TaskSpec, validate_predictions
from training_core.data import load_samples, read_records
from training_core.splits import build_lineage, check_lineage, inspect_splits, validate_lineage_report, validate_splits
from training_tasks.text_classification import TextClassificationTask
from training_tasks.upload.adapter import UploadTask

ROOT = Path(__file__).resolve().parents[1]


def load_example(name, adapter):
    path = ROOT / "configs" / name
    config = TaskSpec.model_validate_json(path.read_text(encoding="utf-8"))
    datasets = {key: load_samples((path.parent / spec.path).resolve(), adapter) for key, spec in config.datasets.items()}
    return config, datasets


def test_public_text_fixture_trains_three_classes_and_survives_reload(tmp_path):
    task = TextClassificationTask()
    config, datasets = load_example("text_demo.json", task)
    validate_splits(inspect_splits(datasets, config.datasets, task.fingerprints), for_training=True)
    training, report = task.prepare(datasets["train"])
    assert report["used"] == 24 and report["excluded"] == 0
    assert Counter(sample.target for sample in training) == {"food": 8, "travel": 8, "technology": 8}
    model = task.fit(training, config)
    assert set(model.classes_) == {"food", "travel", "technology"}
    assert model.structured_features_ == 0
    before = task.predict(model, datasets["development"])
    task.save(model, tmp_path / "model.joblib")
    after = task.predict(task.load(tmp_path / "model.joblib"), datasets["development"])
    validate_predictions(datasets["development"], after)
    assert before == after
    metrics = task.evaluate(datasets["development"], after, config.evaluation_mode)
    assert metrics["total"] == 6
    assert metrics["labels"] == ["food", "technology", "travel"]
    # This is an engineering fixture, not an accuracy benchmark to optimize.
    assert 0 <= metrics["accuracy"] <= 1


def test_heldout_fixture_has_no_cross_partition_overlap_and_uses_saved_lineage(tmp_path):
    task = TextClassificationTask()
    config, datasets = load_example("text_independent.json", task)
    report = inspect_splits(datasets, config.datasets, task.fingerprints)
    validate_splits(report, for_training=True, require_independent=True)
    assert all(not any(pair["overlaps"].values()) for pair in report["pairs"])
    learning_names = ("train", "development")
    lineage = build_lineage({key: datasets[key] for key in learning_names},
                            {key: config.datasets[key] for key in learning_names}, task.fingerprints)
    path = tmp_path / "lineage.json"
    path.write_text(json.dumps(lineage), encoding="utf-8")
    evaluation = check_lineage(datasets["independent"], config.datasets["independent"], task.fingerprints,
                               json.loads(path.read_text(encoding="utf-8")))
    validate_lineage_report(evaluation)
    assert evaluation["independent_test_eligible"]
    assert evaluation["overlaps"] == {"id": 0, "exact": 0, "template": 0, "group": 0}
    assert Counter(sample.target for sample in datasets["independent"]) == {"food": 2, "travel": 2, "technology": 2}


def test_public_upload_fixture_matches_synthetic_generator_and_trains(tmp_path):
    generator = runpy.run_path(str(ROOT / "scripts" / "capture_baseline.py"))["upload_fixture"]
    rows = read_records(ROOT / "examples" / "upload_train.jsonl")
    fields = {"event_id", "label", "request", "response"}
    assert [{key: row[key] for key in fields} for row in rows] == generator()
    assert all(row["source"] == "public_synthetic_upload_fixture_v1" for row in rows)
    task = UploadTask()
    config, datasets = load_example("upload_train.json", task)
    prepared, report = task.prepare(datasets["train"])
    assert report["total"] == 49 and report["used"] == 48 and report["excluded"] == 1
    assert report["reasons"] == {"target:unknown": 1}
    model = task.fit(prepared, config)
    before = task.predict(model, datasets["train"])
    task.save(model, tmp_path / "upload.joblib")
    after = task.predict(task.load(tmp_path / "upload.joblib"), datasets["train"])
    assert after == before
    validate_predictions(datasets["train"], after)
    assert task.evaluate(datasets["train"], after, config.evaluation_mode)["total"] == 49
