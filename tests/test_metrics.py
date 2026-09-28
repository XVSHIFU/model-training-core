from upload_judge.metrics import (
    semantic_confirmed_precision,
    semantic_metrics,
    tdp_compatibility_metrics,
)


def test_semantic_four_class_metrics_and_safety_gate():
    gold = [
        "confirmed_upload_success",
        "likely_upload_success",
        "failed",
        "unknown",
        "failed",
    ]
    predictions = [
        "confirmed_upload_success",
        "likely_upload_success",
        "failed",
        "unknown",
        "likely_upload_success",
    ]
    metrics = semantic_metrics(gold, predictions)
    assert metrics["accuracy_4"] == 0.8
    assert metrics["failed_to_success"] == 1
    assert metrics["hard_gates"]["gold_failed_to_success_eq_0"] is False
    assert semantic_confirmed_precision(gold, predictions) == 1.0


def test_tdp_false_success_uses_full_failed_window(make_event):
    events = [
        make_event('{"success":true,"fileId":"42"}', event_id="step", filename="part.step"),
        make_event('{"success":true,"url":"/uploads/shell.php"}', event_id="php", filename="shell.php"),
        make_event('{"success":false}', event_id="jpg", filename="a.jpg"),
    ]
    labels = ["failed", "failed", "failed"]
    predictions = ["confirmed_upload_success", "likely_upload_success", "failed"]
    metrics = tdp_compatibility_metrics(events, labels, predictions)
    assert metrics["false_success_count"] == 2
    assert metrics["false_success_rate"] == 0.666667


def test_tdp_false_success_boundary_is_inclusive(make_event):
    events = [make_event("{}", event_id=f"event-{index}") for index in range(100)]
    labels = ["failed"] * 100
    predictions = ["likely_upload_success", *(["failed"] * 99)]
    metrics = tdp_compatibility_metrics(events, labels, predictions)
    assert metrics["false_success_rate"] == 0.01
    assert metrics["hard_gates"]["false_success_rate_le_0_01"] is True
