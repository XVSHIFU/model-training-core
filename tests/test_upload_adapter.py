from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from training_core.contracts import Prediction, TaskSpec
from training_tasks.upload import UploadTask
from training_tasks.upload.fingerprints import exact_fingerprint, template_fingerprint
from training_tasks.upload.representation import represent
from upload_judge.decision import Thresholds
from upload_judge.features import build_model_text
from upload_judge.judge import UploadJudge
from upload_judge.metrics import build_evaluation
from upload_judge.ml_model import UploadClassifier


class SpyClassifier:
    thresholds = Thresholds()
    model_version = "adapter-test"

    def __init__(self, probability=0.9):
        self.probability = probability
        self.calls = []

    def predict_success_proba(self, events):
        self.calls.append([event.event_id for event in events])
        return np.asarray([self.probability] * len(events), dtype=float)


def test_parse_preserves_unknown_label_provenance_and_group(make_event):
    task = UploadTask()
    record = make_event().model_dump()
    record.update(
        gold_verdict="unknown", v4_source="reviewed", v3_source="weak", source="raw",
        metadata={"group": "application-a", "review": {"reviewer": "test"}},
    )
    original = deepcopy(record)
    sample = task.parse(record)
    assert sample.target == "unknown"
    assert sample.label_status == "labeled"
    assert sample.source == "reviewed"
    assert sample.group_id == "application-a"
    assert sample.metadata["gold_verdict"] == "unknown"
    assert "gold_verdict" not in sample.inputs.model_dump()
    sample.metadata["review"]["reviewer"] = "changed"
    assert record == original
    unlabeled = task.parse(make_event().model_dump())
    assert unlabeled.target is None
    assert unlabeled.label_status == "unlabeled"


def test_parse_label_precedence_and_event_validation(make_event):
    record = make_event().model_dump()
    record.update(label="failed", gold_verdict="unknown")
    assert UploadTask().parse(record).target == "failed"
    record["label"] = None
    assert UploadTask().parse(record).target == "unknown"
    with pytest.raises(ValidationError):
        UploadTask().parse({"event_id": "missing-request-response"})


@pytest.mark.parametrize("status", ["conflict", "unresolved", "unlabeled"])
def test_explicit_annotation_status_is_preserved_and_excluded_from_training(status, make_event):
    task = UploadTask()
    sample = task.parse({**make_event().model_dump(), "label": "failed", "label_status": status})
    assert sample.target == "failed"
    assert sample.label_status == status
    prepared, report = task.prepare([sample])
    assert prepared == []
    assert report["excluded"] == 1
    assert report["used"] == 0


def test_parse_rejects_invalid_annotation_status(make_event):
    with pytest.raises(ValueError, match="label_status"):
        UploadTask().parse({**make_event().model_dump(), "label_status": "typo"})


def test_training_targets_are_explicit_and_semantic_targets_are_preserved(make_event):
    task = UploadTask()
    labels = ["confirmed_upload_success", "likely_upload_success", "success", "failed", "failure", "unknown", None]
    samples = [task.parse({**make_event(event_id=f"id-{i}").model_dump(), "label": label}) for i, label in enumerate(labels)]
    prepared, report = task.prepare(samples)
    assert [sample.target for sample in prepared] == labels[:5]
    assert [sample.metadata["training_target"] for sample in prepared] == [1, 1, 1, 0, 0]
    assert [sample.target for sample in samples] == labels
    assert samples[-2].label_status == "labeled"
    assert samples[-1].label_status == "unlabeled"
    assert report


def test_all_strong_rules_bypass_actual_model_scoring(make_event):
    task = UploadTask()
    events = [
        make_event('{"success":false}', event_id="failed"),
        make_event('{"success":true,"url":"/uploads/a.jpg"}', filename="a.jpg", event_id="saved"),
    ]
    samples = [task.parse(event.model_dump()) for event in events]
    low, high = SpyClassifier(0.0), SpyClassifier(1.0)
    predictions = task.predict(low, samples)
    assert [prediction.output for prediction in predictions] == [prediction.output for prediction in task.predict(high, samples)]
    # The original judge makes an empty-batch call; it runs no vectorization or
    # classifier prediction for events already resolved by a strong rule.
    assert low.calls == [[]]
    assert high.calls == [[]]
    assert [prediction.output["verdict"] for prediction in predictions] == ["failed", "confirmed_upload_success"]
    assert all(prediction.output["meta"]["source"] == "rule" for prediction in predictions)


def test_predict_preserves_full_legacy_output_count_and_order(make_event):
    task = UploadTask()
    events = [
        make_event('{"success":false}', event_id="z"),
        make_event('{"message":"model candidate"}', event_id="a"),
        make_event('{"success":true,"fileId":"42"}', filename="a.jpg", event_id="m"),
    ]
    samples = [task.parse(event.model_dump()) for event in events]
    classifier = SpyClassifier()
    predictions = task.predict(classifier, samples)
    expected = [result.to_dict() for result in UploadJudge(classifier=SpyClassifier()).judge_batch(events)]
    assert len(predictions) == len(samples)
    assert [prediction.sample_id for prediction in predictions] == ["z", "a", "m"]
    assert [prediction.output for prediction in predictions] == expected
    assert classifier.calls == [["a"]]
    assert task.predict(classifier, []) == []


def test_fingerprints_match_historical_algorithm_even_with_omitted_defaults(make_event):
    path = Path(__file__).resolve().parents[1] / "pipelines" / "build_datasets.py"
    spec = importlib.util.spec_from_file_location("historical_upload_fingerprints", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = [
        make_event("id 42 at 2026-09-28T13:14:15 123e4567-e89b-12d3-a456-426614174000 hash abcdef0123456789").model_dump(),
        {"event_id": "minimal", "request": {}, "response": {}},
        {
            "event_id": "normalized", "request": {"method": "post", "uri": "https://example.test/upload?q=1#ignored"},
            "response": {"status_code": 201, "content_type": " Application/JSON; charset=utf-8 ", "body": "at 1759000000 id 007"},
        },
    ]
    for record in records:
        expected = {"exact": module.exact_fingerprint(record), "template": module.template_fingerprint(record)}
        assert exact_fingerprint(record) == expected["exact"]
        assert template_fingerprint(record) == expected["template"]
        assert UploadTask().fingerprints(UploadTask().parse(record)) == expected
    grouped = UploadTask().parse({**records[0], "group_id": "same-session"})
    assert UploadTask().fingerprints(grouped)["group"] == "same-session"


def test_representation_preserves_legacy_text_and_feature_order(make_event):
    events = [make_event('{"success":true,"url":"/uploads/shell.php"}')]
    texts, rows = represent(events)
    assert texts == [build_model_text(event) for event in events]
    assert rows == [UploadClassifier._struct_row(event) for event in events]
    assert len(rows[0]) == 24


def test_fit_forwards_backend_seed_thresholds_and_original_labels(make_event, monkeypatch):
    from training_tasks.upload import adapter

    class CapturedClassifier:
        def __init__(self, **kwargs):
            self.configuration = kwargs

        def fit(self, events, labels):
            self.events, self.labels = events, labels

    monkeypatch.setattr(adapter, "UploadClassifier", CapturedClassifier)
    task = UploadTask()
    samples = [task.parse({**make_event(event_id=str(i)).model_dump(), "label": label}) for i, label in enumerate(["likely_upload_success", "failed", "unknown"])]
    config = TaskSpec(task_id="upload", backend="logistic", seed=123, parameters={"thresholds": {"confirmed": 0.92, "likely": 0.65, "failed": 0.93}})
    model = task.fit(samples, config)
    assert model.configuration == {"backend": "logistic", "random_state": 123, "thresholds": Thresholds(0.92, 0.65, 0.93)}
    assert model.labels == ["likely_upload_success", "failed"]
    assert [event.event_id for event in model.events] == ["0", "1"]


def test_fit_rejects_unsupported_parameters():
    with pytest.raises(ValueError, match="Unsupported upload parameters"):
        UploadTask().fit([], TaskSpec(task_id="upload", parameters={"learning_rate": 0.01}))


@pytest.mark.parametrize("status,target", [("conflict", "failed"), ("unresolved", "failed"), ("unlabeled", "failed"), ("labeled", None)])
def test_evaluation_rejects_any_non_truth_sample(status, target, make_event):
    task = UploadTask()
    samples = [task.parse({**make_event().model_dump(), "label": target, "label_status": status})]
    predictions = task.predict(SpyClassifier(), samples)
    with pytest.raises(ValueError, match="labeled, resolved targets"):
        task.evaluate(samples, predictions)


def test_evaluation_rejects_empty_data():
    with pytest.raises(ValueError, match="nonempty labeled samples"):
        UploadTask().evaluate([], [])


@pytest.mark.parametrize("mode", ["semantic", "tdp_v2"])
def test_evaluation_reuses_legacy_metrics_and_keeps_unknown(mode, make_event):
    task = UploadTask()
    events = [make_event('{"success":false}', event_id="failed"), make_event('{"note":"not enough evidence"}', event_id="unknown")]
    labels = ["failed", "unknown"]
    samples = [task.parse({**event.model_dump(), "gold_verdict": label}) for event, label in zip(events, labels)]
    predictions = task.predict(SpyClassifier(), samples)
    expected = build_evaluation(events, labels, [prediction.output["verdict"] for prediction in predictions], mode=mode)
    assert task.evaluate(samples, predictions, mode) == {**expected, "excluded_unlabeled": 0}
    with pytest.raises(ValueError, match="IDs/order"):
        task.evaluate(samples, list(reversed(predictions)), mode)
    with pytest.raises(ValueError, match="event_id"):
        task.evaluate(samples, [Prediction(samples[0].sample_id, predictions[1].output), predictions[1]], mode)


def test_joblib_save_load_stays_compatible_with_legacy_classifier(make_event, tmp_path):
    task = UploadTask()
    samples = []
    for index in range(8):
        label = "success" if index % 2 else "failed"
        body = '{"success":true,"url":"/uploads/file.jpg"}' if index % 2 else '{"success":false,"message":"rejected"}'
        event = make_event(body, filename="file.jpg", event_id=str(index))
        samples.append(task.parse({**event.model_dump(), "label": label}))
    model = task.fit(samples, TaskSpec(task_id="upload", backend="logistic"))
    path = tmp_path / "upload.joblib"
    task.save(model, path)
    restored, legacy = task.load(path), UploadClassifier.load(path)
    events = [sample.inputs for sample in samples]
    np.testing.assert_allclose(restored.predict_success_proba(events), legacy.predict_success_proba(events))
    np.testing.assert_allclose(restored.predict_success_proba(events), model.predict_success_proba(events))
