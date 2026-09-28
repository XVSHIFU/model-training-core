from __future__ import annotations

import pytest

from upload_judge.schemas import UploadEvent


@pytest.fixture
def make_event():
    def factory(
        body: str = "",
        *,
        event_id: str = "evt",
        status: int = 200,
        content_type: str = "application/json",
        filename: str = "shell.php",
        ext: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> UploadEvent:
        return UploadEvent.model_validate(
            {
                "event_id": event_id,
                "request": {
                    "method": "POST",
                    "uri": "/api/upload",
                    "headers": {},
                    "content_type": "multipart/form-data",
                    "filename": filename,
                    "file_ext": ext if ext is not None else (filename.rsplit(".", 1)[-1] if "." in filename else ""),
                    "field_name": "file",
                    "body_excerpt": "multipart",
                },
                "response": {
                    "status_code": status,
                    "headers": headers or {},
                    "content_type": content_type,
                    "body": body,
                },
            }
        )
    return factory
