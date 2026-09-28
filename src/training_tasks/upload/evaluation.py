"""Task-owned semantic and legacy TDP outcome evaluation."""
from __future__ import annotations

from typing import Any

from training_core.contracts import Prediction, Sample, validate_predictions
from upload_judge.metrics import build_evaluation
from upload_judge.schemas import JudgmentResult


def evaluate_predictions(
    samples: list[Sample], predictions: list[Prediction], mode: str | None = None
) -> dict[str, Any]:
    validate_predictions(samples, predictions)
    if not samples:
        raise ValueError("Upload evaluation requires nonempty labeled samples")
    if any(sample.label_status != "labeled" or sample.target is None for sample in samples):
        raise ValueError("Upload evaluation requires labeled, resolved targets for every sample")
    evaluation_mode = mode or "tdp_v2"
    if evaluation_mode not in {"semantic", "tdp_v2"}:
        raise ValueError("Upload evaluation mode must be semantic or tdp_v2")
    events, labels, verdicts = [], [], []
    for sample, prediction in zip(samples, predictions):
        result = JudgmentResult.model_validate(prediction.output)
        if result.event_id != sample.sample_id:
            raise ValueError("Upload output event_id must match its sample_id")
        events.append(sample.inputs)
        label = str(sample.target)
        # The training alias is also accepted by binary TDP evaluation. The
        # semantic evaluator deliberately requires the four semantic labels.
        labels.append("failed" if evaluation_mode == "tdp_v2" and label == "failure" else label)
        verdicts.append(result.verdict.value)
    report = build_evaluation(events, labels, verdicts, mode=evaluation_mode)
    report["excluded_unlabeled"] = len(samples) - len(events)
    return report
