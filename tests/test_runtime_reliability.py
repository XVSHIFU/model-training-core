import hashlib
import json
from pathlib import Path

import pytest

from training_core import artifacts
from training_core.runner import Workflow, load_config
from training_tasks.registry import resolve_task

ROOT = Path(__file__).resolve().parents[1]


def test_failed_serialization_does_not_destroy_previous_state(tmp_path):
    path = tmp_path / "run.json"
    artifacts.write_json(path, {"status": "running"})
    original = path.read_bytes()
    for invalid in ({"value": object()}, {"value": float("nan")}):
        with pytest.raises((TypeError, ValueError)):
            artifacts.write_json(path, invalid, overwrite=True)
        assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_failed_atomic_replace_keeps_last_readable_state(tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    artifacts.write_json(path, {"status": "running"})

    def occupied(*args):
        raise PermissionError("file held by another process")

    monkeypatch.setattr(artifacts.os, "replace", occupied)
    with pytest.raises(PermissionError):
        artifacts.write_json(path, {"status": "succeeded"}, overwrite=True)
    assert json.loads(path.read_text()) == {"status": "running"}
    assert list(tmp_path.iterdir()) == [path]


def test_initial_publication_does_not_overwrite_a_concurrent_writer(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    actual_link = artifacts.os.link

    def concurrent_write(source, target):
        target.write_text('"concurrent"', encoding="utf-8")
        actual_link(source, target)

    monkeypatch.setattr(artifacts.os, "link", concurrent_write)
    with pytest.raises(FileExistsError):
        artifacts.write_json(path, {"owned": "this run"})
    assert json.loads(path.read_text()) == "concurrent"
    assert list(tmp_path.iterdir()) == [path]


def demo_config(tmp_path):
    config = load_config(ROOT / "configs/text_demo.json")
    config.run_root = str(tmp_path / "runs")
    return config


def test_user_interrupt_is_recorded_and_preserved(tmp_path):
    config = demo_config(tmp_path)
    task = resolve_task(config.task_id)

    def interrupt(*args):
        raise KeyboardInterrupt()

    task.fit = interrupt
    with pytest.raises(KeyboardInterrupt):
        Workflow(lambda _: task).train(config)
    record = next(Path(config.run_root).glob("*/run.json"))
    assert json.loads(record.read_text())["status"] == "interrupted"
    assert not (record.parent / "manifest.json").exists()


@pytest.mark.parametrize("state", ["running", "failed", "interrupted", "missing"])
def test_only_successful_training_runs_can_be_loaded(tmp_path, state):
    artifact = Path(Workflow(resolve_task).train(demo_config(tmp_path))["artifact"])
    record = artifact / "run.json"
    if state == "missing":
        record.unlink()
    else:
        value = json.loads(record.read_text())
        value["status"] = state
        record.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="completion record|successfully committed"):
        artifacts.load_manifest(artifact)


def test_training_records_the_bytes_it_loaded_and_a_configuration_snapshot(tmp_path):
    config = demo_config(tmp_path)
    train = tmp_path / "train.jsonl"
    original = Path(config.datasets["train"].path).read_bytes()
    train.write_bytes(original)
    config.datasets["train"].path = str(train)
    task = resolve_task(config.task_id)
    original_parse = task.parse

    def mutate_external_inputs(record):
        # Simulate an editor writing the next batch while the current one runs.
        train.write_text('{"next_batch": true}', encoding="utf-8")
        config.seed = 12345
        return original_parse(record)

    task.parse = mutate_external_inputs
    result = Workflow(lambda _: task).train(config)
    run = Path(result["artifact"])
    assert result["training_summary"]["used"] == 24
    assert json.loads((run / "input_hashes.json").read_text())["train"] == hashlib.sha256(original).hexdigest()
    assert json.loads((run / "config.json").read_text())["seed"] == 42
    assert task.load(run / "model").random_state == 42
