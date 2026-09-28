import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from pydantic import BaseModel

from training_core.artifacts import load_manifest
from training_core.contracts import DatasetSpec, Prediction, Sample, validate_predictions
from training_core.runner import Workflow, load_config
from training_tasks.registry import resolve_task

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    value = load_config(ROOT / "configs" / "text_demo.json")
    value.run_root = str(tmp_path / "runs")
    # Training must not open a final test, even if that path is unavailable.
    value.datasets["sealed"] = DatasetSpec(path=str(tmp_path / "must-not-read.json"),
                                           purpose="independent_test", independent=True, provenance="sealed")
    return value


def test_train_portable_artifact_and_independence_enforcement(config, tmp_path):
    workflow = Workflow(resolve_task)
    trained = workflow.train(config)
    artifact = Path(trained["artifact"])
    assert trained["training_summary"]["class_counts"] == {"food": 8, "technology": 8, "travel": 8}
    manifest, model_path = load_manifest(artifact)
    model = resolve_task("text_classification").load(model_path)
    assert model.classes_.tolist() == ["food", "technology", "travel"]
    assert manifest["training_summary"]["used"] == 24
    moved = tmp_path / "moved-artifact"
    shutil.copytree(artifact, moved)
    # Original locations, including the sealed missing path, aren't needed.
    predicted = workflow.predict(ROOT / "examples" / "text_dev.jsonl", artifact=moved, run_root=tmp_path / "predictions")
    assert predicted["count"] == 6
    records = [json.loads(line) for line in Path(predicted["predictions"]).read_text(encoding="utf-8").splitlines()]
    assert all(set(row) == {"sample_id", "output", "status"} for row in records)
    spec = DatasetSpec(path=str(ROOT / "examples" / "text_independent.jsonl"), purpose="independent_test",
                       independent=True, provenance="held-out synthetic engineering fixture")
    evaluated = workflow.evaluate(spec, artifact=moved, run_root=tmp_path / "evaluation")
    assert evaluated["independent_test_eligible"] is True
    assert evaluated["metrics"]["total"] == 6
    spec.path = config.datasets["train"].path
    with pytest.raises(RuntimeError, match="lineage_overlap"):
        workflow.evaluate(spec, artifact=moved, run_root=tmp_path / "rejected")
    failed = next((tmp_path / "rejected").glob("*/run.json"))
    assert json.loads(failed.read_text())["status"] == "failed"
    spec.purpose = "historical_regression"
    regression = workflow.evaluate(spec, artifact=moved, run_root=tmp_path / "history")
    assert regression["independent_test_eligible"] is False
    assert regression["metrics"]["total"] == 24


def test_failed_training_leaves_record(config):
    config.parameters = {"made_up_algorithm": True}
    with pytest.raises(RuntimeError, match="Unsupported text parameters"):
        Workflow(resolve_task).train(config)
    records = list(Path(config.run_root).glob("*/run.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["status"] == "failed"
    assert record["details"]["error_type"] == "ValueError"
    assert not (records[0].parent / "manifest.json").exists()


def test_artifact_integrity_precedes_loading(config):
    trained = Workflow(resolve_task).train(config)
    artifact = Path(trained["artifact"])
    (artifact / "model").write_bytes(b"corrupt model")
    with pytest.raises(ValueError, match="integrity"):
        load_manifest(artifact)


def test_model_must_be_in_inventory(config):
    artifact = Path(Workflow(resolve_task).train(config)["artifact"])
    path = artifact / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"].pop("model")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fully covered"):
        load_manifest(artifact)


def test_manifest_argument_protects_entire_artifact_directory(config):
    workflow = Workflow(resolve_task)
    artifact = Path(workflow.train(config)["artifact"])
    with pytest.raises(ValueError, match="protected"):
        workflow.predict(ROOT / "examples" / "text_dev.jsonl", artifact=artifact / "manifest.json", run_root=artifact / "nested-runs")
    assert not (artifact / "nested-runs").exists()


def test_development_use_cannot_be_renamed_independent(config, tmp_path):
    workflow = Workflow(resolve_task)
    artifact = Path(workflow.train(config)["artifact"])
    spec = DatasetSpec(path=str(ROOT / "examples" / "text_independent.jsonl"), purpose="development")
    workflow.evaluate(spec, artifact=artifact, run_root=tmp_path / "development")
    moved = tmp_path / "moved-with-history"
    shutil.copytree(artifact, moved)
    spec.purpose = "independent_test"
    spec.independent = True
    spec.provenance = "incorrectly reclassified development data"
    with pytest.raises(RuntimeError, match="lineage_overlap"):
        workflow.evaluate(spec, artifact=moved, run_root=tmp_path / "independent")
    # An ordinary repeated frozen independent evaluation remains permissible.
    fresh = Path(workflow.train(config)["artifact"])
    for _ in range(2):
        assert workflow.evaluate(spec, artifact=fresh, run_root=tmp_path / "repeat")["independent_test_eligible"]


@pytest.mark.parametrize("damage", ["model", "manifest"])
def test_bad_artifact_records_failed_run(config, tmp_path, damage):
    workflow = Workflow(resolve_task)
    artifact = Path(workflow.train(config)["artifact"])
    if damage == "model":
        (artifact / "model").write_bytes(b"corrupt")
    else:
        path = artifact / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest.pop("model_path")
        path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError):
        workflow.predict(ROOT / "examples" / "text_dev.jsonl", artifact=artifact, run_root=tmp_path / "bad")
    record = json.loads(next((tmp_path / "bad").glob("*/run.json")).read_text())
    assert record["status"] == "failed"
    assert record["details"]["error_type"] == "ValueError"


def test_cli_real_roundtrip(tmp_path):
    value = load_config(ROOT / "configs" / "text_demo.json")
    value.run_root = str(tmp_path / "runs")
    config_path = tmp_path / "config.json"
    config_path.write_text(value.model_dump_json(), encoding="utf-8")
    base = [sys.executable, "-m", "training_tasks.registry"]
    checked = subprocess.run([*base, "validate-data", "--config", str(config_path)], capture_output=True, text=True, encoding="utf-8")
    assert checked.returncode == 0, checked.stderr
    trained = subprocess.run([*base, "train", "--config", str(config_path)], capture_output=True, text=True, encoding="utf-8")
    assert trained.returncode == 0, trained.stderr
    artifact = json.loads(trained.stdout)["artifact"]
    predicted = subprocess.run([*base, "predict", "--artifact", artifact, "--input", str(ROOT / "examples" / "text_dev.jsonl"),
                                "--run-root", str(tmp_path / "prediction-runs")], capture_output=True, text=True, encoding="utf-8")
    assert predicted.returncode == 0, predicted.stderr
    assert json.loads(predicted.stdout)["count"] == 6


def test_core_has_no_business_imports():
    for path in (ROOT / "src" / "training_core").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(name.startswith(("upload_judge", "training_tasks", "training_backends")) for name in names), path
    result = subprocess.run([sys.executable, "-c", "import training_core.runner, sys; assert not any(k.startswith('upload_judge') for k in sys.modules)"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_contract_accepts_task_validated_text_and_coordinates():
    class TextOutput(BaseModel):
        text: str

    class CoordinatesOutput(BaseModel):
        boxes: list[tuple[float, float, float, float]]

    sample = Sample("one", inputs={})
    for output in (TextOutput(text="an answer"), CoordinatesOutput(boxes=[(0, 0, 10, 20)])):
        predictions = [Prediction("one", output.model_dump())]
        validate_predictions([sample], predictions)
        assert "confidence" not in predictions[0].output
    with pytest.raises(ValueError, match="IDs/order"):
        validate_predictions([sample], [Prediction("wrong", {})])
