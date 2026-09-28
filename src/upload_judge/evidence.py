"""确定性证据提取。"""

from upload_judge.features import extract_structured_features
from upload_judge.schemas import Evidence, UploadEvent

RESOURCE_EVIDENCE_TYPES = {"returned_path", "returned_file_id", "returned_filename", "created_location", "filename_echo"}


def extract_evidence(event: UploadEvent) -> tuple[list[Evidence], list[Evidence]]:
    f = extract_structured_features(event)
    positive = [Evidence(type="success_field", value=value) for value in f["success_fields"]]
    negative = [Evidence(type="failure_field", value=value) for value in f["fail_fields"]]
    positive.extend(Evidence(type="returned_path", value=value) for value in f["paths"])
    positive.extend(Evidence(type="returned_file_id", value=value) for value in f["file_ids"])
    positive.extend(Evidence(type="returned_filename", value=value) for value in f["returned_filenames"])
    if f["status_code"] == 201 and f["location"]:
        positive.append(Evidence(type="created_location", value=f["location"]))
    if f["filename_in_response"] and f["filename"]:
        positive.append(Evidence(type="filename_echo", value=f["filename"]))
    if f["body_empty"]:
        negative.append(Evidence(type="ambiguous_signal", value="empty_response_body"))
    return _dedupe(positive), _dedupe(negative)


def has_resource_evidence(items: list[Evidence]) -> bool:
    return any(item.type in RESOURCE_EVIDENCE_TYPES for item in items)


def has_success_semantics(items: list[Evidence]) -> bool:
    return any(item.type == "success_field" for item in items)


def has_failure_semantics(items: list[Evidence]) -> bool:
    return any(item.type == "failure_field" for item in items)


def _dedupe(items: list[Evidence]) -> list[Evidence]:
    seen: set[tuple[str, str]] = set()
    result: list[Evidence] = []
    for item in items:
        key = (item.type, item.value)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
