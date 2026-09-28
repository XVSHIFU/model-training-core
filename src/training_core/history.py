"""Serialize local artifact evaluation and record use before model execution.

The journal stores hashed sample membership, not sample content. A reservation
remains after failure or process interruption because the data may have already
been used. Locks are for local Windows/Linux filesystems; network filesystem and
power-loss durability guarantees are outside this module's scope.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import errno
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Callable, Iterator
import uuid

from training_core.artifacts import guard_output, write_json
from training_core.contracts import DatasetSpec, Sample
from training_core.data import stable_hash
from training_core.splits import build_lineage, check_lineage, validate_lineage

_PURPOSES = {"development", "historical_regression", "independent_test"}
_BUSY_ERRORS = {errno.EACCES, errno.EAGAIN, errno.EDEADLK}


class EvaluationRejected(ValueError):
    """An eligibility failure whose report should be saved with the failed run."""

    def __init__(self, report: dict):
        self.report = deepcopy(report)
        super().__init__("Evaluation data is ineligible: " + ", ".join(report["violations"]))


def _history_directory(artifact_root: Path) -> Path:
    root = Path(artifact_root).resolve()
    if not root.is_dir():
        raise ValueError("Evaluation history requires an existing artifact directory")
    history = root / "evaluation_history"
    if history.resolve() != history:
        raise ValueError("Evaluation history must remain inside the artifact")
    guard_output(history, allow_existing=True)
    history.mkdir(exist_ok=True)
    if not history.is_dir():
        raise ValueError("Evaluation history must be a directory")
    return history


@contextmanager
def _file_lock(history: Path, timeout: float) -> Iterator[None]:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
        raise ValueError("Evaluation lock timeout must be a finite nonnegative number")
    lock_path = history / ".lock"
    if lock_path.resolve() != lock_path:
        raise ValueError("Evaluation lock must remain inside the history directory")
    guard_output(lock_path, allow_existing=True)
    # Windows byte-range locks may extend beyond EOF, so the empty lock file
    # needs no racing initialization write. Never delete this persistent inode.
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    acquired = False
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError as exc:
                if exc.errno not in _BUSY_ERRORS:
                    raise OSError("Unable to acquire artifact evaluation lock") from exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for artifact evaluation lock") from exc
                time.sleep(min(0.05, remaining))
        yield
    finally:
        active_error = sys.exc_info()[1]
        try:
            if acquired:
                try:
                    if os.name == "nt":
                        import msvcrt
                        os.lseek(descriptor, 0, os.SEEK_SET)
                        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError as exc:
                    if active_error is None:
                        raise
                    if hasattr(active_error, "add_note"):
                        active_error.add_note(f"Evaluation lock cleanup also failed: {type(exc).__name__}")
        finally:
            os.close(descriptor)


def _validated_record(record: dict) -> dict:
    if not isinstance(record, dict) or type(record.get("version")) is not int or record["version"] != 1:
        raise ValueError("Invalid artifact evaluation history version")
    allowed = {"version", "run_id", "purpose", "provenance", "provenance_sha256", "lineage"}
    if set(record) - allowed:
        raise ValueError("Unrecognized artifact evaluation history fields")
    if not isinstance(record.get("run_id"), str) or not record["run_id"].strip():
        raise ValueError("Invalid artifact evaluation history run ID")
    if record.get("purpose") not in _PURPOSES:
        raise ValueError("Invalid artifact evaluation history purpose")
    if "provenance" in record and not isinstance(record["provenance"], str):
        raise ValueError("Invalid artifact evaluation history provenance")
    if "provenance_sha256" in record:
        value = record["provenance_sha256"]
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("Invalid artifact evaluation history provenance hash")
    document = record.get("lineage")
    if not isinstance(document, dict) or type(document.get("version")) is not int:
        raise ValueError("Missing or invalid artifact evaluation history lineage")
    entries = validate_lineage(document)
    if not entries:
        raise ValueError("Artifact evaluation history must contain membership entries")
    if any(entry["purpose"] != record["purpose"] or entry["count"] <= 0 for entry in entries.values()):
        raise ValueError("Artifact evaluation history purpose/count is inconsistent")
    return entries


def _combined_lineage(history: Path, lineage: dict) -> dict:
    entries = deepcopy(validate_lineage(lineage))
    for path in sorted(history.glob("*.json")):
        if path.resolve().parent != history or not path.is_file():
            raise ValueError("Evaluation history record escapes its directory or is not a file")
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Unreadable artifact evaluation history record") from exc
        # Validate independent records too, before deciding whether their prior
        # use disqualifies a subsequent frozen independent evaluation.
        prior_entries = _validated_record(record)
        for name, entry in prior_entries.items():
            if entry["purpose"] != "independent_test":
                entries[f"evaluation:{path.stem}:{name}"] = entry
    return {"version": 1, "entries": entries}


@contextmanager
def evaluation_session(
    artifact_root: Path,
    lineage: dict | None,
    samples: list[Sample],
    spec: DatasetSpec,
    fingerprints: Callable[[Sample], dict[str, str]],
    *,
    run_id: str,
    timeout: float = 30.0,
) -> Iterator[dict]:
    """Check and reserve evaluation use, holding its local lock through metrics.

    Callers must perform model loading/prediction/evaluation inside the context.
    Rejected eligibility raises ``EvaluationRejected`` with a portable report.
    Models without lineage stay read-only and cannot gain independent status.
    """
    if spec.purpose not in _PURPOSES:
        raise ValueError("Evaluation purpose must be development, historical_regression or independent_test")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Evaluation run ID must be a nonempty string")
    if not samples:
        raise ValueError("Evaluation requires nonempty samples")
    if lineage is None:
        report = check_lineage(samples, spec, fingerprints, None)
        if not report["eligible"]:
            raise EvaluationRejected(report)
        yield report
        return
    history = _history_directory(artifact_root)
    with _file_lock(history, timeout):
        combined = _combined_lineage(history, lineage)
        report = check_lineage(samples, spec, fingerprints, combined)
        if not report["eligible"]:
            raise EvaluationRejected(report)
        usage = {
            "version": 1, "run_id": run_id, "purpose": spec.purpose,
            "provenance_sha256": stable_hash(spec.provenance),
            "lineage": build_lineage({"evaluation": samples}, {"evaluation": spec}, fingerprints),
        }
        _validated_record(usage)
        # The atomic record must exist before control reaches model execution.
        # A failure here prevents use; failure after yield keeps the reservation.
        write_json(history / (uuid.uuid4().hex + ".json"), usage)
        yield report
