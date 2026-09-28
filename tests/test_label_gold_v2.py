from pipelines.label_gold import stratified_sample


def _item(event_id: str, body: str, ext: str) -> dict:
    filename = f"sample.{ext}" if ext else "sample"
    return {
        "event_id": event_id,
        "label": "failed",
        "request": {
            "method": "POST",
            "uri": "/upload",
            "headers": {},
            "content_type": "multipart/form-data",
            "filename": filename,
            "file_ext": ext,
            "field_name": "file",
            "body_excerpt": "multipart",
        },
        "response": {
            "status_code": 200,
            "headers": {},
            "content_type": "application/json",
            "body": body,
        },
    }


def test_v2_sampling_is_deterministic_without_extension_requirement():
    rows = [_item(f"event-{index}", f'{{"message":"item-{index}"}}', "jpg") for index in range(340)]
    first = stratified_sample(rows, 300, seed=42)
    second = stratified_sample(rows, 300, seed=42)
    assert len(first) == 300
    assert [item["event_id"] for item in first] == [item["event_id"] for item in second]
