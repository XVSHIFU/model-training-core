"""Historical upload dataset fingerprints, independent of pipeline scripts.

These intentionally retain the exact normalization and field selection from
``pipelines/build_datasets.py``. They are compatibility checks, not a complete
definition of semantic duplication.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from upload_judge.convert import normalize_uri

_HEX = re.compile(r"\b[0-9a-fA-F]{16,}\b")
_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_TIMESTAMP = re.compile(r"\b(?:1[5-9]\d{8}|2\d{9}|\d{4}-\d{1,2}-\d{1,2}(?:[T ]\d{1,2}:\d{2}:\d{2})?)\b")
_NUMBER = re.compile(r"\b\d+\b")


def exact_fingerprint(item: dict[str, Any]) -> str:
    req, resp = item["request"], item["response"]
    value = "|".join(
        [
            str(req.get("method", "")).upper(),
            normalize_uri(req.get("uri", "")),
            str(resp.get("status_code", 0)),
            str(resp.get("content_type", "")).lower().split(";", 1)[0].strip(),
            str(resp.get("body", "")),
        ]
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def template_fingerprint(item: dict[str, Any]) -> str:
    req, resp = item["request"], item["response"]
    body = str(resp.get("body", ""))
    body = _UUID.sub("<uuid>", body)
    body = _HEX.sub("<hex>", body)
    body = _TIMESTAMP.sub("<timestamp>", body)
    body = _NUMBER.sub("<num>", body)
    value = "|".join(
        [
            str(req.get("method", "")).upper(),
            normalize_uri(req.get("uri", "")),
            str(resp.get("status_code", 0)),
            str(resp.get("content_type", "")).lower().split(";", 1)[0].strip(),
            body,
        ]
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
