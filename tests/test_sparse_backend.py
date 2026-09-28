"""Backend contract tests use synthetic non-upload data and isolated artifacts."""

from __future__ import annotations

import ast
from pathlib import Path

import joblib
import numpy as np
import pytest

from training_backends.sparse_classifier import SparseClassifier


def classification_data():
    texts, targets, features = [], [], []
    for index, (label, text) in enumerate([
        ("garden", "garden flower soil watering roots"),
        ("finance", "finance invoice payment account budget"),
        ("travel", "travel train ticket station journey"),
    ]):
        for number in range(24):
            texts.append(f"{text} record {number}")
            targets.append(label)
            features.append([float(index), float(number % 2)])
    return texts, targets, features


@pytest.mark.parametrize("backend", ["logistic", "lightgbm"])
def test_non_upload_three_classes_roundtrip(tmp_path, backend):
    texts, targets, features = classification_data()
    model = SparseClassifier(backend, classifier_params={"n_jobs": 1})
    report = model.fit(texts, targets, features)
    assert model.classes_.tolist() == ["finance", "garden", "travel"]
    assert report["class_counts"] == [24, 24, 24]
    assert report["training_samples"] == len(targets)
    assert model.structured_features_ == 2
    assert model.n_features_in_ == len(model.vectorizer.vocabulary_) + 2
    expected = model.predict_proba(texts, features)
    assert expected.shape == (len(texts), 3)
    # LogisticRegression may calculate its softmax with float32 features.
    np.testing.assert_allclose(expected.sum(axis=1), 1.0, atol=1e-6)
    assert np.mean(model.predict(texts, features) == targets) > 0.95

    artifact = tmp_path / "nested" / "classifier.joblib"
    model.save(artifact)
    restored = SparseClassifier.load(artifact)
    assert restored.vectorizer.vocabulary_ == model.vectorizer.vocabulary_
    assert restored.manifest == model.manifest
    np.testing.assert_array_equal(restored.predict_proba(texts, features), expected)
    np.testing.assert_array_equal(restored.predict(texts, features), model.predict(texts, features))


def test_prediction_cannot_refit_vocabulary_or_idf(monkeypatch):
    texts, targets, _ = classification_data()
    model = SparseClassifier("logistic", classifier_params={"n_jobs": 1})
    model.fit(texts, targets)
    vocabulary = dict(model.vectorizer.vocabulary_)
    idf = model.vectorizer.idf_.copy()

    def fail_if_called(*args, **kwargs):
        pytest.fail("Prediction attempted to fit the vectorizer")

    monkeypatch.setattr(model.vectorizer, "fit", fail_if_called)
    monkeypatch.setattr(model.vectorizer, "fit_transform", fail_if_called)
    model.predict_proba(["zzzz previously unseen heldout material"])
    assert model.vectorizer.vocabulary_ == vocabulary
    np.testing.assert_array_equal(model.vectorizer.idf_, idf)
    assert model.structured_features_ == 0


@pytest.mark.parametrize("structured", [None, [[1.0]], [[1.0, 2.0, 3.0]], [[np.nan, 0]], [[np.inf, 0]], [[1, 2], [3, 4]]])
def test_prediction_rejects_wrong_or_missing_structured_features(structured):
    texts, targets, features = classification_data()
    model = SparseClassifier("logistic", classifier_params={"n_jobs": 1})
    model.fit(texts, targets, features)
    with pytest.raises(ValueError):
        model.predict_proba(["garden"], structured)


@pytest.mark.parametrize("structured", [[[0], [1, 2]], [[0]], [[float("nan")], [1]], [["invalid"], [0]]])
def test_training_rejects_invalid_structured_matrix(structured):
    with pytest.raises(ValueError):
        SparseClassifier("logistic").fit(["flower", "ticket"], ["garden", "travel"], structured)


@pytest.mark.parametrize("targets", [
    ["garden", None], ["garden", ""], ["garden", "  "],
    ["garden", 1], [0.1, 0.2], [float("nan"), 1],
    [["garden"], ["travel"]], [{"label": "garden"}, {"label": "travel"}],
    ["garden", "garden"],
])
def test_training_rejects_invalid_class_targets(targets):
    with pytest.raises(ValueError):
        SparseClassifier("logistic").fit(["flower", "ticket"], targets)


def test_no_structured_features_and_empty_prediction():
    texts, targets, _ = classification_data()
    model = SparseClassifier("logistic", classifier_params={"n_jobs": 1})
    model.fit(texts, targets)
    assert model.predict_proba([]).shape == (0, 3)
    assert model.predict([]).shape == (0,)
    with pytest.raises(ValueError, match="width"):
        model.predict(["garden"], [[1]])
    with pytest.raises(RuntimeError, match="not fitted"):
        SparseClassifier().predict_proba([])


def test_load_rejects_inconsistent_feature_dimensions(tmp_path):
    texts, targets, features = classification_data()
    model = SparseClassifier("logistic", classifier_params={"n_jobs": 1})
    model.fit(texts, targets, features)
    path = tmp_path / "malformed.joblib"
    model.save(path)
    payload = joblib.load(path)
    payload["structured_features"] += 1
    joblib.dump(payload, path)
    with pytest.raises(ValueError, match="dimensions"):
        SparseClassifier.load(path)


def test_backend_does_not_import_upload_modules():
    import training_backends.sparse_classifier as backend_module

    tree = ast.parse(Path(backend_module.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name.startswith(("upload_judge", "training_tasks")) for name in imports)


def test_upload_uses_shared_backend_and_legacy_artifact(tmp_path, make_event, monkeypatch):
    from upload_judge.ml_model import UploadClassifier

    events = [
        make_event('{"success":true,"url":"/files/report.pdf"}', filename="report.pdf"),
        make_event('{"success":true,"url":"/files/budget.pdf"}', filename="budget.pdf"),
        make_event('{"success":false,"message":"forbidden"}', status=403),
        make_event('{"success":false,"message":"invalid file type"}'),
        make_event("pending review"),
    ]
    labels = ["confirmed_upload_success", "likely_upload_success", "failed", "failed", "unknown"]
    fit_calls = []
    backend_fit = SparseClassifier.fit

    def record_fit(self, texts, targets, structured=None):
        fit_calls.append((list(targets), len(structured[0])))
        return backend_fit(self, texts, targets, structured)

    monkeypatch.setattr(SparseClassifier, "fit", record_fit)
    model = UploadClassifier(backend="logistic")
    manifest = model.fit(events, labels)
    assert fit_calls == [([1, 1, 0, 0], 24)]
    assert manifest["training_samples"] == 4
    assert manifest["training_success"] == manifest["training_failed"] == 2
    assert manifest["structured_features"] == 24
    assert model.classifier.C == 2.0
    assert model.classifier.max_iter == 1000
    expected = model.predict_success_proba(events)
    artifact = tmp_path / "upload.joblib"
    model.save(artifact)
    assert set(joblib.load(artifact)) == {
        "backend", "thresholds", "random_state", "vectorizer", "classifier", "model_version", "manifest"
    }
    restored = UploadClassifier.load(artifact)
    np.testing.assert_array_equal(restored.predict_success_proba(events), expected)
    assert restored.predict_success_proba([]).shape == (0,)
