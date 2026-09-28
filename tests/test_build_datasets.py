from pipelines.build_datasets import (
    exact_fingerprint,
    leakage_report,
    split_gold_tune_regress,
    template_fingerprint,
)


def _item(body, event_id="x", label="failed"):
    return {
        "event_id": event_id,
        "label": label,
        "request": {"method": "POST", "uri": "https://example.test/upload", "headers": {}, "content_type": "", "filename": "a.php", "file_ext": "php", "field_name": "file", "body_excerpt": ""},
        "response": {"status_code": 200, "headers": {}, "content_type": "application/json", "body": body},
    }


def test_exact_and_template_fingerprints():
    first = _item('{"id":123,"hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}')
    second = _item('{"id":456,"hash":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}')
    assert exact_fingerprint(first) != exact_fingerprint(second)
    assert template_fingerprint(first) == template_fingerprint(second)


def test_leakage_report_detects_exact_overlap():
    item = _item("same")
    report = leakage_report({"train": [item], "val": [], "final_test": [dict(item)]})
    assert report["exact_overlaps"]["train__final_test"] == 1


def test_gold_split_is_deterministic_stratified_and_template_safe():
    rows = []
    labels = ["confirmed_upload_success", "likely_upload_success", "failed", "unknown"]
    for label in labels:
        for index in range(5):
            marker = chr(ord("a") + index) * (index + 1)
            item = _item(f'{{"label":"{label}","marker":"{marker}"}}', event_id=f"{label}-{index}")
            item["gold_verdict"] = label
            rows.append(item)
    first_tune, first_regress = split_gold_tune_regress(rows, tune_size=12, seed=7)
    second_tune, second_regress = split_gold_tune_regress(rows, tune_size=12, seed=7)
    assert [item["event_id"] for item in first_tune] == [item["event_id"] for item in second_tune]
    assert [item["event_id"] for item in first_regress] == [item["event_id"] for item in second_regress]
    assert len(first_tune) == 12
    assert {template_fingerprint(item) for item in first_tune}.isdisjoint(
        {template_fingerprint(item) for item in first_regress}
    )
