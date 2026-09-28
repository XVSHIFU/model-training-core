"""Preserve the complete rule -> model -> evidence decision chain."""
from __future__ import annotations

from typing import Any

from upload_judge.judge import UploadJudge
from upload_judge.schemas import UploadEvent


def judge_events(classifier: Any, events: list[UploadEvent]) -> list[dict[str, Any]]:
    return [result.to_dict() for result in UploadJudge(classifier=classifier).judge_batch(events)]
