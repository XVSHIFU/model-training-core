"""Task-specific text and 24 numeric features used by the legacy artifact."""
from __future__ import annotations

from upload_judge.features import build_model_text
from upload_judge.ml_model import UploadClassifier
from upload_judge.schemas import UploadEvent


def model_text(event: UploadEvent) -> str:
    return build_model_text(event)


def structured_row(event: UploadEvent) -> list[float]:
    # Keep a single implementation so existing joblib feature order is stable.
    return UploadClassifier._struct_row(event)


def represent(events: list[UploadEvent]) -> tuple[list[str], list[list[float]]]:
    return [model_text(event) for event in events], [structured_row(event) for event in events]
