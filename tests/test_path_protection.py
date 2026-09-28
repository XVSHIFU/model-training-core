"""All fixtures are temporary; these tests never write original project assets."""

from __future__ import annotations

import json

import joblib
import pytest

from training_backends.sparse_classifier import SparseClassifier
from training_core import artifacts
from upload_judge import cli
from upload_judge.ml_model import UploadClassifier


@pytest.fixture
def protected_workspace(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    original = tmp_path / "original-deliverable"
    (root / "local").mkdir(parents=True)
    original.mkdir()
    (root / "local" / "protected.json").write_text(
        json.dumps({"paths": [str(original)]}), encoding="utf-8"
    )
    monkeypatch.setattr(artifacts, "workspace_root", lambda: root)
    monkeypatch.delenv("MODEL_TRAINING_PROTECTED_PATHS", raising=False)
    return root, original


def fitted_model(kind, make_event):
    if kind == "sparse":
        model = SparseClassifier("logistic")
        model.fit(["garden flowers", "garden flowers", "train tickets", "train tickets"], ["garden", "garden", "travel", "travel"])
    else:
        model = UploadClassifier("logistic")
        model.fit(
            [make_event('{"success":true}'), make_event('{"success":true}'),
             make_event('{"success":false}'), make_event('{"success":false}')],
            ["success", "success", "failed", "failed"],
        )
    return model


@pytest.mark.parametrize("kind", ["sparse", "upload"])
@pytest.mark.parametrize("protected_name", ["models", "data", "reports", ".git", ".venv", "original"])
def test_save_rejects_protected_directory_before_mkdir(kind, protected_name, protected_workspace, make_event):
    root, original = protected_workspace
    parent = (original if protected_name == "original" else root / protected_name) / "new-subdirectory"
    model = fitted_model(kind, make_event)
    with pytest.raises(ValueError, match="protected asset"):
        model.save(parent / "candidate.joblib")
    assert not parent.exists()


@pytest.mark.parametrize("kind", ["sparse", "upload"])
def test_save_allows_new_run_artifact_but_never_overwrites(kind, protected_workspace, make_event):
    root, _ = protected_workspace
    output = root / "runs" / "test-run" / "model.joblib"
    model = fitted_model(kind, make_event)
    model.save(output)
    original_hash = artifacts.sha256_file(output)
    with pytest.raises(FileExistsError, match="overwrite"):
        model.save(output)
    assert artifacts.sha256_file(output) == original_hash
    assert type(model).load(output).is_fitted


@pytest.mark.parametrize("collision", ["model", "manifest", "protected"])
def test_train_checks_model_and_manifest_before_reading_or_fitting(collision, protected_workspace, monkeypatch, capsys):
    root, original = protected_workspace
    output = root / "runs" / "train" / "candidate.joblib"
    if collision == "protected":
        output = original / "candidate.joblib"
    else:
        output.parent.mkdir(parents=True)
        existing = output if collision == "model" else output.with_suffix(".manifest.json")
        existing.write_bytes(b"preserve-existing-artifact")

    def must_not_run(*args, **kwargs):
        pytest.fail("Training/data loading started before output preflight")

    monkeypatch.setattr(cli, "load_events", must_not_run)
    monkeypatch.setattr(cli.UploadClassifier, "fit", must_not_run)
    assert cli.main(["train", "-d", "absent-input.json", "-o", str(output)]) == 1
    assert "Refusing" in capsys.readouterr().err
    if collision != "protected":
        assert existing.read_bytes() == b"preserve-existing-artifact"
    if collision != "model":
        assert not output.exists()


@pytest.mark.parametrize("command", ["evaluate", "judge", "convert", "convert-alert", "verify-acceptance"])
def test_other_old_cli_writes_are_preflighted(command, protected_workspace, monkeypatch, capsys):
    root, original = protected_workspace
    output = str(original / "must-not-create.json")
    calls = []

    def must_not_run(*args, **kwargs):
        calls.append(True)
        pytest.fail("Processing started despite protected output")

    for name in ("load_events", "convert_file", "convert_http_dump", "run_acceptance"):
        monkeypatch.setattr(cli, name, must_not_run)
    arguments = {
        "evaluate": ["-d", "absent.json", "-m", "absent.joblib", "--json-out", output],
        "judge": ["-i", "absent.json", "-m", "absent.joblib", "-o", output],
        "convert": ["-i", "absent.jsonl", "--label", "failed", "-o", output],
        "convert-alert": ["-i", "absent.txt", "-o", output],
        "verify-acceptance": ["-m", "absent.joblib", "--gold", "gold.json", "--tdp", "tdp.json", "--out", output, "--json-out", str(root / "runs" / "report.json"), "--smoke"],
    }
    assert cli.main([command, *arguments[command]]) == 1
    assert "protected asset" in capsys.readouterr().err
    assert not calls
    assert not (original / "must-not-create.json").exists()


def test_multi_output_evaluation_rejects_second_collision_before_any_output(protected_workspace, capsys):
    root, _ = protected_workspace
    report = root / "runs" / "evaluation" / "old.md"
    report.parent.mkdir(parents=True)
    report.write_text("preserve", encoding="utf-8")
    fresh_json = report.with_suffix(".json")
    assert cli.main(["evaluate", "-d", "absent.json", "-m", "absent.joblib", "--json-out", str(fresh_json), "--md-out", str(report)]) == 1
    assert "overwrite" in capsys.readouterr().err
    assert report.read_text(encoding="utf-8") == "preserve"
    assert not fresh_json.exists()


def test_formal_acceptance_guards_hidden_sentinel_before_running(protected_workspace, monkeypatch, capsys):
    root, _ = protected_workspace
    run = root / "runs" / "acceptance"
    run.mkdir(parents=True)
    sentinel = run / ".acceptance_v2_consumed.json"
    sentinel.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(cli, "run_acceptance", lambda *a, **k: pytest.fail("Acceptance should not run"))
    assert cli.main(["verify-acceptance", "-m", "model.joblib", "--gold", "gold.json", "--tdp", "tdp.json", "--out", str(run / "report.md"), "--json-out", str(run / "report.json")]) == 1
    assert "overwrite" in capsys.readouterr().err
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_evaluation_outputs_must_be_distinct(protected_workspace, capsys):
    root, _ = protected_workspace
    output = root / "runs" / "evaluation.json"
    assert cli.main(["evaluate", "-d", "absent.json", "-m", "model.joblib", "--json-out", str(output), "--md-out", str(output)]) == 1
    assert "distinct" in capsys.readouterr().err
    assert not output.exists()


def test_old_train_cli_writes_fresh_model_and_original_manifest(protected_workspace, make_event, capsys):
    root, _ = protected_workspace
    source = root / "fixture.json"
    rows = []
    for index in range(8):
        success = bool(index % 2)
        event = make_event('{"success":true}' if success else '{"success":false}', event_id=str(index))
        rows.append({**event.model_dump(), "label": "success" if success else "failed"})
    source.write_text(json.dumps(rows), encoding="utf-8")
    output = root / "runs" / "fresh-training" / "candidate.joblib"
    arguments = ["train", "-d", str(source), "-o", str(output), "--model-type", "b1"]
    assert cli.main(arguments) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["saved"] == str(output)
    assert summary["training_samples"] == 8
    assert set(joblib.load(output)) == {"backend", "thresholds", "random_state", "vectorizer", "classifier", "model_version", "manifest"}
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_success"] == manifest["training_failed"] == 4
    assert UploadClassifier.load(output).is_fitted
    previous_hash = artifacts.sha256_file(output)
    assert cli.main(arguments) == 1
    assert artifacts.sha256_file(output) == previous_hash
