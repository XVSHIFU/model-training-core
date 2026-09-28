"""Local-process history safety; all records and models are temporary fixtures."""
from __future__ import annotations

from copy import deepcopy
import json
import multiprocessing
from pathlib import Path

import pytest

from training_core import history
from training_core.contracts import DatasetSpec, Sample
from training_core.data import stable_hash
from training_core.history import EvaluationRejected, evaluation_session
from training_core.splits import build_lineage


def fingerprints(sample):
    return {"exact": stable_hash(sample.inputs), "template": stable_hash(sample.inputs)}


def training_lineage():
    return build_lineage(
        {"train": [Sample("training-id", {"text": "training content"}, "class", label_status="labeled")]},
        {"train": DatasetSpec(path="unused.json", purpose="train")}, fingerprints,
    )


def evaluation_samples():
    return [Sample("private-evaluation-id", {"text": "private evaluation content"}, "class", group_id="private-group", label_status="labeled")]


def evaluation_spec(purpose="development"):
    return DatasetSpec(path="unused.json", purpose=purpose, independent=purpose == "independent_test", provenance="private collection description")


def legacy_record(purpose="development"):
    return {
        "version": 1, "run_id": "previous-run", "purpose": purpose,
        "provenance": "legacy fixture", "lineage": build_lineage(
            {"evaluation": evaluation_samples()}, {"evaluation": evaluation_spec(purpose)}, fingerprints
        ),
    }


def install_record(root, record):
    folder = root / "evaluation_history"
    folder.mkdir(exist_ok=True)
    path = folder / "prior.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def process_evaluation(root, purpose, started, entered, release, results):
    """Spawn-safe worker exercises the same public session as Workflow."""
    started.set()
    try:
        with evaluation_session(Path(root), training_lineage(), evaluation_samples(), evaluation_spec(purpose), fingerprints, run_id=f"worker-{purpose}", timeout=5):
            entered.set()
            if not release.wait(10):
                raise TimeoutError("Test worker was not released")
        results.put(("passed", purpose))
    except BaseException as exc:
        results.put((type(exc).__name__, purpose))


def stop_worker(process, release):
    if process.is_alive():
        release.set()
    process.join(10)
    if process.is_alive():
        process.terminate()
        process.join(5)


def test_reservation_exists_before_model_use_and_contains_only_hashed_membership(tmp_path):
    samples = evaluation_samples()
    with evaluation_session(tmp_path, training_lineage(), samples, evaluation_spec(), fingerprints, run_id="current-run") as report:
        assert report["eligible"]
        paths = list((tmp_path / "evaluation_history").glob("*.json"))
        assert len(paths) == 1
        text = paths[0].read_text(encoding="utf-8")
        record = json.loads(text)
        assert record["purpose"] == "development"
        assert record["lineage"]["entries"]["evaluation"]["count"] == 1
        for private in (samples[0].sample_id, samples[0].inputs["text"], samples[0].group_id, evaluation_spec().provenance):
            assert private not in text
        assert record["provenance_sha256"] == stable_hash(evaluation_spec().provenance)


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_failure_or_interrupt_preserves_use_and_releases_lock(tmp_path, exception):
    with pytest.raises(exception):
        with evaluation_session(tmp_path, training_lineage(), evaluation_samples(), evaluation_spec(), fingerprints, run_id="failed-run"):
            raise exception("Failure after use started")
    with pytest.raises(EvaluationRejected, match="lineage_overlap") as raised:
        with evaluation_session(tmp_path, training_lineage(), evaluation_samples(), evaluation_spec("independent_test"), fingerprints, run_id="next-run", timeout=0):
            pytest.fail("Previously used data must not reach model evaluation")
    assert raised.value.report["independent_test_eligible"] is False
    assert raised.value.report["overlaps"]["exact"] == 1


def test_repeat_independent_is_allowed_but_every_record_is_validated(tmp_path):
    for index in range(2):
        with evaluation_session(tmp_path, training_lineage(), evaluation_samples(), evaluation_spec("independent_test"), fingerprints, run_id=f"repeat-{index}") as report:
            assert report["independent_test_eligible"]
    assert len(list((tmp_path / "evaluation_history").glob("*.json"))) == 2


def test_legacy_v1_record_without_coverage_still_blocks_reclassification(tmp_path):
    record = legacy_record()
    for entry in record["lineage"]["entries"].values():
        entry.pop("coverage", None)
    install_record(tmp_path, record)
    with pytest.raises(EvaluationRejected, match="lineage_overlap"):
        with evaluation_session(tmp_path, training_lineage(), evaluation_samples(), evaluation_spec("independent_test"), fingerprints, run_id="new-run"):
            pytest.fail("Legacy development use must still be honored")


@pytest.mark.parametrize("damage", ["missing_lineage", "invalid_count", "invalid_hash", "mismatched_purpose", "empty_entries", "boolean_version", "unknown_field"])
def test_malformed_independent_records_fail_before_any_new_use(tmp_path, damage):
    record = legacy_record("independent_test")
    if damage == "missing_lineage":
        record.pop("lineage")
    elif damage == "invalid_count":
        record["lineage"]["entries"]["evaluation"]["count"] = "broken"
    elif damage == "invalid_hash":
        record["lineage"]["entries"]["evaluation"]["keys"]["exact"] = ["raw data"]
    elif damage == "mismatched_purpose":
        record["purpose"] = "development"
    elif damage == "empty_entries":
        record["lineage"]["entries"] = {}
    elif damage == "boolean_version":
        record["version"] = True
    else:
        record["ignored_bad_data"] = "must not silently accept"
    install_record(tmp_path, record)
    with pytest.raises(ValueError):
        with evaluation_session(tmp_path, training_lineage(), evaluation_samples(), evaluation_spec("independent_test"), fingerprints, run_id="new-run"):
            pytest.fail("Malformed history must block execution")
    assert len(list((tmp_path / "evaluation_history").glob("*.json"))) == 1


def test_truncated_json_blocks_evaluation(tmp_path):
    path = install_record(tmp_path, legacy_record())
    path.write_text('{"version": 1,', encoding="utf-8")
    with pytest.raises(ValueError, match="Unreadable"):
        with evaluation_session(tmp_path, training_lineage(), evaluation_samples(), evaluation_spec(), fingerprints, run_id="new-run"):
            pytest.fail("Unreadable history must block execution")


def test_failed_registration_prevents_execution_and_does_not_leave_lock_held(tmp_path, monkeypatch):
    original = history.write_json

    def fail_write(*args, **kwargs):
        raise OSError("injected registration failure")

    monkeypatch.setattr(history, "write_json", fail_write)
    with pytest.raises(OSError, match="registration"):
        with evaluation_session(tmp_path, training_lineage(), evaluation_samples(), evaluation_spec(), fingerprints, run_id="failed-write"):
            pytest.fail("Model must not run before registration succeeds")
    monkeypatch.setattr(history, "write_json", original)
    with evaluation_session(tmp_path, training_lineage(), evaluation_samples(), evaluation_spec(), fingerprints, run_id="retry", timeout=0):
        pass


def test_legacy_model_without_lineage_remains_read_only(tmp_path):
    model_file = tmp_path / "legacy-model"
    model_file.write_bytes(b"placeholder")
    with evaluation_session(model_file, None, evaluation_samples(), evaluation_spec("historical_regression"), fingerprints, run_id="legacy") as report:
        assert not report["independent_test_eligible"]
    assert not (tmp_path / "evaluation_history").exists()
    with pytest.raises(EvaluationRejected, match="missing_training_lineage"):
        with evaluation_session(model_file, None, evaluation_samples(), evaluation_spec("independent_test"), fingerprints, run_id="legacy-independent"):
            pytest.fail("Legacy model without lineage cannot claim independence")


def test_development_and_independent_evaluation_serialize_across_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    started, entered, release = context.Event(), context.Event(), context.Event()
    independent_started, independent_entered, independent_release = context.Event(), context.Event(), context.Event()
    results = context.Queue()
    development = context.Process(target=process_evaluation, args=(str(tmp_path), "development", started, entered, release, results))
    independent = context.Process(target=process_evaluation, args=(str(tmp_path), "independent_test", independent_started, independent_entered, independent_release, results))
    development.start()
    try:
        assert entered.wait(10), "Development worker did not obtain its lock"
        independent.start()
        assert independent_started.wait(10)
        assert not independent_entered.wait(0.2)
        release.set()
        development.join(10)
        independent.join(10)
        assert development.exitcode == independent.exitcode == 0
        assert {results.get(timeout=3), results.get(timeout=3)} == {("passed", "development"), ("EvaluationRejected", "independent_test")}
        assert not independent_entered.is_set()
    finally:
        stop_worker(development, release)
        if independent.pid is not None:
            stop_worker(independent, independent_release)


def test_lock_timeout_is_explicit_and_crashed_process_releases_os_lock(tmp_path):
    context = multiprocessing.get_context("spawn")
    started, entered, release = context.Event(), context.Event(), context.Event()
    results = context.Queue()
    worker = context.Process(target=process_evaluation, args=(str(tmp_path), "development", started, entered, release, results))
    worker.start()
    try:
        assert entered.wait(10)
        with pytest.raises(TimeoutError, match="evaluation lock"):
            with evaluation_session(tmp_path, training_lineage(), evaluation_samples(), evaluation_spec("historical_regression"), fingerprints, run_id="blocked", timeout=0.1):
                pytest.fail("A concurrent evaluator must not pass the lock")
        worker.terminate()
        worker.join(10)
        with pytest.raises(EvaluationRejected, match="lineage_overlap"):
            with evaluation_session(tmp_path, training_lineage(), evaluation_samples(), evaluation_spec("independent_test"), fingerprints, run_id="after-crash", timeout=1):
                pytest.fail("Crash must retain development use")
    finally:
        stop_worker(worker, release)


@pytest.mark.parametrize("target", ["history", "lock", "record"])
def test_symlink_escape_is_rejected(tmp_path, target):
    root, outside = tmp_path / "artifact", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    history_path = root / "evaluation_history"
    try:
        if target == "history":
            history_path.symlink_to(outside, target_is_directory=True)
        else:
            history_path.mkdir()
            external = outside / "external.json"
            external.write_text(json.dumps(legacy_record()), encoding="utf-8")
            (history_path / (".lock" if target == "lock" else "escaped.json")).symlink_to(external)
    except OSError:
        pytest.skip("This environment does not allow creating symlinks")
    before = {path.name: path.read_bytes() for path in outside.iterdir() if path.is_file()}
    with pytest.raises(ValueError, match="inside|escapes"):
        with evaluation_session(root, training_lineage(), evaluation_samples(), evaluation_spec(), fingerprints, run_id="escape"):
            pytest.fail("History must remain within its artifact")
    assert {path.name: path.read_bytes() for path in outside.iterdir() if path.is_file()} == before
