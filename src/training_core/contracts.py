"""Small, task-independent contracts. Outputs are validated by task adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

PURPOSES = {"train", "development", "historical_regression", "independent_test"}


@dataclass
class Sample:
    sample_id: str
    inputs: Any
    target: Any = None
    source: str = ""
    group_id: str | None = None
    label_status: str = "unlabeled"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    sample_id: str
    output: Any
    status: str = "predicted"


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    purpose: Literal["train", "development", "historical_regression", "independent_test"]
    provenance: str = ""
    # Administrative attestation is necessary but never replaces overlap checks.
    independent: bool = False


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    backend: str = "logistic"
    seed: int = 42
    datasets: dict[str, DatasetSpec] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    run_root: str = "runs"
    protected_paths: list[str] = Field(default_factory=list)
    artifact_path: str | None = None
    evaluation_mode: str | None = None


class TaskAdapter(Protocol):
    task_id: str

    def parse(self, record: dict[str, Any]) -> Sample: ...
    def fingerprints(self, sample: Sample) -> dict[str, str]: ...
    def prepare(self, samples: list[Sample]) -> tuple[list[Sample], dict[str, Any]]: ...
    def fit(self, samples: list[Sample], config: TaskSpec) -> Any: ...
    def predict(self, model: Any, samples: list[Sample]) -> list[Prediction]: ...
    def save(self, model: Any, path: Path) -> None: ...
    def load(self, path: Path) -> Any: ...
    def evaluate(self, samples: list[Sample], predictions: list[Prediction], mode: str | None = None) -> dict[str, Any]: ...


def validate_predictions(samples: list[Sample], predictions: list[Prediction]) -> None:
    if len(samples) != len(predictions):
        raise ValueError("Prediction count must equal input count")
    if [s.sample_id for s in samples] != [p.sample_id for p in predictions]:
        raise ValueError("Prediction IDs/order must match inputs")
