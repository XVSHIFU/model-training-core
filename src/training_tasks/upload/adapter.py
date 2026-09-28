"""Adapt upload outcomes to the reusable training contracts."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from training_core.contracts import Prediction, Sample, TaskSpec, validate_predictions
from training_core.data import prepare_targets
from training_tasks.upload.evaluation import evaluate_predictions
from training_tasks.upload.fingerprints import exact_fingerprint, template_fingerprint
from training_tasks.upload.policy import judge_events
from upload_judge.decision import Thresholds
from upload_judge.ml_model import UploadClassifier
from upload_judge.schemas import UploadEvent

TARGET_MAPPING = {
    "confirmed_upload_success": 1,
    "likely_upload_success": 1,
    "success": 1,
    "failed": 0,
    "failure": 0,
}


class UploadTask:
    task_id = "upload"

    def parse(self, record: dict[str, Any]) -> Sample:
        event = UploadEvent.model_validate(record)
        # Match legacy dataset loading: label takes precedence over gold_verdict.
        # A missing/null label can still fall back to a supplied gold annotation.
        target = record.get("label")
        if target is None:
            target = record.get("gold_verdict")
        if target is not None and not isinstance(target, str):
            raise ValueError("Upload label/gold_verdict must be a string or null")
        metadata = deepcopy(record.get("metadata") or {})
        if not isinstance(metadata, dict):
            raise ValueError("Upload metadata must be an object")
        for key in ("label", "gold_verdict", "v4_source", "v3_source", "source"):
            if key in record:
                metadata[key] = deepcopy(record[key])
        # Compute before schema defaults so omitted fields match the historical
        # fingerprint algorithm exactly; only hashes are retained as metadata.
        metadata["_upload_fingerprints"] = {
            "exact": exact_fingerprint(record),
            "template": template_fingerprint(record),
        }
        group = record.get("group_id", metadata.get("group_id", metadata.get("group")))
        source = next((record[key] for key in ("v4_source", "v3_source", "source") if record.get(key)), "")
        label_status = record.get("label_status", "unlabeled" if target is None else "labeled")
        if label_status not in {"labeled", "unlabeled", "unresolved", "conflict"}:
            raise ValueError("Unrecognized upload label_status")
        return Sample(
            sample_id=event.event_id,
            inputs=event,
            target=target,
            source=str(source),
            group_id=str(group) if group is not None else None,
            label_status=label_status,
            metadata=metadata,
        )

    def fingerprints(self, sample: Sample) -> dict[str, str]:
        cached = sample.metadata.get("_upload_fingerprints")
        if cached is not None:
            result = dict(cached)
        else:
            item = sample.inputs.model_dump()
            result = {"exact": exact_fingerprint(item), "template": template_fingerprint(item)}
        if sample.group_id is not None:
            result["group"] = sample.group_id
        return result

    def prepare(self, samples: list[Sample]) -> tuple[list[Sample], dict[str, Any]]:
        # unknown is a labeled semantic outcome, excluded only from this binary
        # fitting objective. Original targets remain available for evaluation.
        return prepare_targets(samples, TARGET_MAPPING, {"unknown"})

    def fit(self, samples: list[Sample], config: TaskSpec) -> UploadClassifier:
        if config.task_id != self.task_id:
            raise ValueError("UploadTask requires task_id='upload'")
        parameters = config.parameters
        unexpected = set(parameters) - {"thresholds"}
        if unexpected:
            raise ValueError(f"Unsupported upload parameters: {sorted(unexpected)}")
        thresholds = Thresholds(**parameters.get("thresholds", {}))
        thresholds.validate()
        classifier = UploadClassifier(
            backend=config.backend, thresholds=thresholds, random_state=config.seed
        )
        prepared, _ = self.prepare(samples)
        classifier.fit([sample.inputs for sample in prepared], [sample.target for sample in prepared])
        return classifier

    def predict(self, model: Any, samples: list[Sample]) -> list[Prediction]:
        outputs = judge_events(model, [sample.inputs for sample in samples])
        predictions = [Prediction(sample_id=output["event_id"], output=output) for output in outputs]
        validate_predictions(samples, predictions)
        return predictions

    def save(self, model: UploadClassifier, path: Path) -> None:
        model.save(path)

    def load(self, path: Path) -> UploadClassifier:
        return UploadClassifier.load(path)

    def evaluate(
        self, samples: list[Sample], predictions: list[Prediction], mode: str | None = None
    ) -> dict[str, Any]:
        return evaluate_predictions(samples, predictions, mode)
