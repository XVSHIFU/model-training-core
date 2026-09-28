"""A small real classification adapter used to prove non-upload reuse."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from training_backends.sparse_classifier import SparseClassifier
from training_core.contracts import Prediction, Sample, TaskSpec
from training_core.data import prepare_targets, stable_hash


class TextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    features: list[Annotated[float, Field(strict=True, allow_inf_nan=False)]] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text input must not be blank")
        return value


class TextClassificationTask:
    task_id = "text_classification"

    def parse(self, record: dict[str, Any]) -> Sample:
        inputs = TextInput.model_validate(record.get("inputs", {"text": record.get("text"), "features": record.get("features", [])}))
        target = record.get("target", record.get("label"))
        if target is not None and (not isinstance(target, str) or not target.strip()):
            raise ValueError("Text class labels must be nonempty strings")
        return Sample(record.get("sample_id", record.get("id")), inputs.model_dump(), target,
                      record.get("source", ""), record.get("group_id"),
                      record.get("label_status", "labeled" if target is not None else "unlabeled"),
                      record.get("metadata", {}))

    def fingerprints(self, sample):
        text = sample.inputs["text"]
        return {"exact": stable_hash(sample.inputs), "template": stable_hash(re.sub(r"\d+", "<N>", " ".join(text.lower().split()))), "group": sample.group_id or ""}

    def prepare(self, samples):
        mapping = {sample.target: sample.target for sample in samples if sample.target is not None}
        return prepare_targets(samples, mapping, set())

    def fit(self, samples, config: TaskSpec):
        allowed = {"vectorizer", "classifier"}
        if set(config.parameters) - allowed:
            raise ValueError(f"Unsupported text parameters: {sorted(set(config.parameters) - allowed)}")
        model = SparseClassifier(backend=config.backend, random_state=config.seed,
                                 vectorizer_params=config.parameters.get("vectorizer"),
                                 classifier_params=config.parameters.get("classifier"))
        model.fit([s.inputs["text"] for s in samples], [s.metadata["training_target"] for s in samples], [s.inputs["features"] for s in samples])
        return model

    def predict(self, model, samples):
        labels = model.predict([s.inputs["text"] for s in samples], [s.inputs["features"] for s in samples])
        return [Prediction(s.sample_id, {"label": str(label)}) for s, label in zip(samples, labels)]

    def save(self, model, path: Path):
        model.save(path)

    def load(self, path: Path):
        return SparseClassifier.load(path)

    def evaluate(self, samples, predictions, mode=None):
        if mode not in (None, "classification"):
            raise ValueError(f"Unsupported evaluation mode: {mode}")
        if not samples or any(s.target is None or s.label_status != "labeled" for s in samples):
            raise ValueError("Evaluation requires nonempty labeled data")
        truth = [s.target for s in samples]
        predicted = [p.output["label"] for p in predictions]
        labels = sorted(set(truth) | set(predicted))
        return {"total": len(samples), "accuracy": accuracy_score(truth, predicted),
                "labels": labels, "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
                "per_class": classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0)}
