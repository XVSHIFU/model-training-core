"""输入、输出与证据 schema。"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    CONFIRMED_UPLOAD_SUCCESS = "confirmed_upload_success"
    LIKELY_UPLOAD_SUCCESS = "likely_upload_success"
    FAILED = "failed"
    UNKNOWN = "unknown"


class RequestInfo(BaseModel):
    method: str = "POST"
    uri: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    content_type: str = ""
    filename: str = ""
    file_ext: str = ""
    field_name: str = ""
    body_excerpt: str = ""


class ResponseInfo(BaseModel):
    status_code: int = 0
    headers: dict[str, str] = Field(default_factory=dict)
    content_type: str = ""
    body: str = ""


class UploadEvent(BaseModel):
    event_id: str
    schema_version: str = "2.0.0"
    request: RequestInfo
    response: ResponseInfo


class Evidence(BaseModel):
    type: str
    value: str


class JudgmentResult(BaseModel):
    event_id: str
    verdict: Verdict
    upload_saved: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    negative_evidence: list[Evidence] = Field(default_factory=list)
    reason: str = ""
    rule_version: str = ""
    model_version: str = ""
    feature_version: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = self.model_dump()
        value["verdict"] = self.verdict.value
        return value
