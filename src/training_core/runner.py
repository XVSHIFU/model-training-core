"""Backend-neutral orchestration with explicit data use and durable run records."""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
from typing import Callable
import uuid

from training_core.artifacts import (
    REPOSITORY_ROOT, begin_run, finish_run, guard_output, inventory,
    load_manifest, runtime_versions, sha256_file, write_json,
)
from training_core.contracts import DatasetSpec, TaskAdapter, TaskSpec, validate_predictions
from training_core.data import load_samples, stable_hash
from training_core.splits import (
    build_lineage, check_lineage, inspect_splits, validate_lineage_report, validate_splits,
)


def load_config(path: str | Path) -> TaskSpec:
    path = Path(path).resolve()
    config = TaskSpec.model_validate_json(path.read_text(encoding="utf-8-sig"))

    def absolute(value):
        candidate = Path(value)
        return str((path.parent / candidate).resolve())

    config.run_root = absolute(config.run_root)
    config.protected_paths = [absolute(value) for value in config.protected_paths]
    for spec in config.datasets.values():
        spec.path = absolute(spec.path)
    if config.artifact_path:
        config.artifact_path = absolute(config.artifact_path)
    return config


def code_fingerprint() -> dict:
    """Content identity also works before the first commit or in a dirty tree."""
    source = Path(__file__).resolve().parents[1]
    files = {p.relative_to(source).as_posix(): sha256_file(p) for p in sorted(source.rglob("*.py"))}
    return {"sha256": stable_hash(files), "files": files}


class Workflow:
    def __init__(self, resolve_task: Callable[[str], TaskAdapter]):
        self.resolve_task = resolve_task

    def _run(self, command, config, operation, *, input_paths=()):
        # Inputs themselves and an explicitly configured legacy model are always
        # protected, even when the caller did not register a local protected root.
        protected = [*config.protected_paths, *input_paths]
        protected.extend(spec.path for spec in config.datasets.values())
        if config.artifact_path:
            protected.append(config.artifact_path)
        run = begin_run(Path(config.run_root), command, config.model_dump(), protected)
        try:
            write_json(run / "environment.json", runtime_versions())
            result = operation(run)
            finish_run(run, status="succeeded", details=result)
            return {"status": "succeeded", "command": command, "run_dir": str(run), **result}
        except Exception as exc:
            finish_run(run, status="failed", details={"error_type": type(exc).__name__, "error": str(exc)})
            raise RuntimeError(f"{command} failed; record: {run / 'run.json'}; {exc}") from exc

    def validate_data(self, config: TaskSpec):
        def operation(run):
            task = self.resolve_task(config.task_id)
            if not config.datasets:
                raise ValueError("No datasets configured")
            datasets = {name: load_samples(spec.path, task) for name, spec in config.datasets.items()}
            summary = {name: task.prepare(rows)[1] for name, rows in datasets.items()}
            write_json(run / "data_summary.json", summary)
            checks = inspect_splits(datasets, config.datasets, task.fingerprints)
            write_json(run / "split_report.json", checks)
            write_json(run / "input_hashes.json", {name: sha256_file(Path(spec.path)) for name, spec in config.datasets.items()})
            validate_splits(checks)
            return {"datasets": len(datasets), "data_summary": summary, "split_report": "split_report.json"}
        return self._run("validate-data", config, operation)

    def train(self, config: TaskSpec):
        def operation(run):
            task = self.resolve_task(config.task_id)
            specs = {name: spec for name, spec in config.datasets.items() if spec.purpose in {"train", "development"}}
            training = [name for name, spec in specs.items() if spec.purpose == "train"]
            if len(training) != 1:
                raise ValueError("Configure exactly one training dataset; merge sources explicitly before training")
            # Final test paths are deliberately never opened by this command.
            datasets = {name: load_samples(spec.path, task) for name, spec in specs.items()}
            checks = inspect_splits(datasets, specs, task.fingerprints)
            write_json(run / "split_report.json", checks)
            validate_splits(checks, for_training=True)
            prepared, summary = task.prepare(datasets[training[0]])
            write_json(run / "data_summary.json", {"training": summary})
            if not prepared:
                raise ValueError("No usable supervised training samples")
            model = task.fit(prepared, config)
            model_path = guard_output(run / "model", config.protected_paths)
            task.save(model, model_path)
            if not model_path.exists():
                raise ValueError("Task adapter did not save a model")
            write_json(run / "lineage.json", build_lineage(datasets, specs, task.fingerprints))
            write_json(run / "input_hashes.json", {name: sha256_file(Path(spec.path)) for name, spec in specs.items()})
            development = {}
            for name, spec in specs.items():
                if spec.purpose == "development":
                    predictions = task.predict(model, datasets[name])
                    validate_predictions(datasets[name], predictions)
                    development[name] = task.evaluate(datasets[name], predictions, config.evaluation_mode)
            write_json(run / "development_metrics.json", {"purpose": "development", "datasets": development})
            files = inventory(model_path, run)
            for name in ("config.json", "lineage.json", "input_hashes.json", "environment.json", "data_summary.json", "split_report.json", "development_metrics.json"):
                files[name] = sha256_file(run / name)
            write_json(run / "manifest.json", {
                "format_version": 1, "task_id": task.task_id, "backend": config.backend,
                "model_path": "model", "config_path": "config.json", "lineage_path": "lineage.json",
                "files": files, "code": code_fingerprint(), "environment": runtime_versions(),
                "training_summary": summary,
                "scope": "Engineering artifact; task-specific quality requires a separate evaluation",
            })
            return {"artifact": str(run), "training_summary": summary, "development_metrics": development}
        return self._run("train", config, operation)

    def _model_context(self, artifact=None, config=None):
        if artifact is not None:
            artifact = Path(artifact).resolve()
            manifest, model_path = load_manifest(artifact)
            root = artifact if artifact.is_dir() else artifact.parent
            saved = load_config(root / manifest["config_path"])
            if saved.task_id != manifest["task_id"] or saved.backend != manifest["backend"]:
                raise ValueError("Artifact task/backend disagrees with saved configuration")
            lineage = json.loads((root / manifest["lineage_path"]).read_text(encoding="utf-8"))
            # A moved artifact must not depend on the old training machine's paths.
            config = saved.model_copy(update={"datasets": {}, "run_root": str(REPOSITORY_ROOT / "runs"),
                                               "artifact_path": None, "protected_paths": []})
            return config, model_path, lineage, root
        if config is None or not config.artifact_path:
            raise ValueError("Provide --artifact or a config containing artifact_path")
        return config, Path(config.artifact_path), None, Path(config.artifact_path)

    @staticmethod
    def _request_config(artifact, config, run_root):
        request = config.model_copy(deep=True) if config else TaskSpec(
            task_id="artifact_pending", run_root=str(REPOSITORY_ROOT / "runs"),
            artifact_path=str(Path(artifact).resolve()) if artifact else None,
        )
        if run_root:
            request.run_root = str(Path(run_root).resolve())
        return request

    @staticmethod
    def _artifact_root_hint(artifact):
        path = Path(artifact).resolve()
        return path if path.is_dir() else path.parent if path.is_file() or path.name == "manifest.json" else path

    @staticmethod
    def _usage_lineage(artifact_root, lineage):
        if lineage is None:
            return None
        combined = deepcopy(lineage)
        history = artifact_root / "evaluation_history"
        if history.resolve().parent != artifact_root.resolve():
            raise ValueError("Evaluation history must remain inside the artifact")
        for path in sorted(history.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("version") != 1:
                raise ValueError("Invalid artifact evaluation history")
            for name, entry in record["lineage"]["entries"].items():
                # Repeating a frozen independent evaluation is reproducibility,
                # whereas any declared development/regression use disqualifies it.
                if entry["purpose"] != "independent_test":
                    combined["entries"][f"evaluation:{path.stem}:{name}"] = entry
        return combined

    def predict(self, input_path, *, artifact=None, config=None, run_root=None):
        request = self._request_config(artifact, config, run_root)
        input_path = Path(input_path).resolve()

        def operation(run):
            effective, model_path, _, artifact_root = self._model_context(artifact, config)
            effective.run_root = request.run_root
            write_json(run / "config.json", effective.model_dump(), overwrite=True)
            task = self.resolve_task(effective.task_id)
            samples = load_samples(input_path, task)
            model = task.load(model_path)
            predictions = task.predict(model, samples)
            validate_predictions(samples, predictions)
            self._write_predictions(run, predictions)
            write_json(run / "inputs.json", {"input_sha256": sha256_file(input_path), "artifact": str(artifact_root)})
            return {"count": len(predictions), "predictions": str(run / "predictions.jsonl")}
        protected = [input_path, self._artifact_root_hint(artifact)] if artifact else [input_path]
        return self._run("predict", request, operation, input_paths=protected)

    def evaluate(self, spec: DatasetSpec, *, artifact=None, config=None, run_root=None, mode=None):
        if spec.purpose == "train":
            raise ValueError("Evaluation purpose must be development, historical_regression or independent_test")
        request = self._request_config(artifact, config, run_root)

        def operation(run):
            effective, model_path, lineage, artifact_root = self._model_context(artifact, config)
            effective.run_root = request.run_root
            write_json(run / "config.json", effective.model_dump(), overwrite=True)
            task = self.resolve_task(effective.task_id)
            samples = load_samples(spec.path, task)
            write_json(run / "evaluation_spec.json", spec.model_dump())
            checks = check_lineage(samples, spec, task.fingerprints, self._usage_lineage(artifact_root, lineage))
            write_json(run / "lineage_report.json", checks)
            validate_lineage_report(checks)
            model = task.load(model_path)
            predictions = task.predict(model, samples)
            validate_predictions(samples, predictions)
            metrics = task.evaluate(samples, predictions, mode or effective.evaluation_mode)
            if lineage is not None:
                # New artifacts own a portable append-only usage journal. Legacy
                # model files are read-only and cannot claim independent lineage.
                usage = {"version": 1, "run_id": run.name, "purpose": spec.purpose,
                         "provenance": spec.provenance,
                         "lineage": build_lineage({"evaluation": samples}, {"evaluation": spec}, task.fingerprints)}
                write_json(artifact_root / "evaluation_history" / (uuid.uuid4().hex + ".json"), usage)
            self._write_predictions(run, predictions)
            result = {"purpose": spec.purpose, "independent_test_eligible": checks["independent_test_eligible"],
                      "provenance": spec.provenance, "data_sha256": sha256_file(Path(spec.path)), "metrics": metrics}
            write_json(run / "metrics.json", result)
            return result
        protected = [spec.path, self._artifact_root_hint(artifact)] if artifact else [spec.path]
        return self._run("evaluate", request, operation, input_paths=protected)

    @staticmethod
    def _write_predictions(run, predictions):
        target = guard_output(run / "predictions.jsonl")
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            for prediction in predictions:
                stream.write(json.dumps(asdict(prediction), ensure_ascii=False, allow_nan=False) + "\n")
