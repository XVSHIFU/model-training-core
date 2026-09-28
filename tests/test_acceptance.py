from upload_judge.acceptance import _acceptance_markdown, _validate_formal_inputs
from upload_judge.metrics import semantic_metrics, tdp_compatibility_metrics


def test_formal_input_check_has_no_extension_quota(make_event):
    classes = ["confirmed_upload_success", "likely_upload_success", "failed", "unknown"]
    gold_events = [
        make_event("{}", event_id=f"gold-{label}-{index}", filename="sample.jpg")
        for label in classes
        for index in range(30)
    ]
    gold_labels = [label for label in classes for _ in range(30)]
    tdp_events = [make_event("{}", event_id=f"tdp-{index}") for index in range(3000)]
    tdp_labels = ["confirmed_upload_success"] * 2700 + ["failed"] * 300
    result = _validate_formal_inputs(gold_events, gold_labels, tdp_events, tdp_labels)
    assert result["passed"] is True
    assert set(result["checks"]) == {
        "tdp_total_ge_3000",
        "tdp_labels_binary",
        "gold_all_four_classes",
        "gold_each_class_ge_30",
        "gold_deducted_from_tdp",
    }


def test_acceptance_report_uses_full_window_tdp_and_single_output(make_event):
    semantic_labels = [
        "confirmed_upload_success",
        "likely_upload_success",
        "failed",
        "unknown",
    ]
    semantic = semantic_metrics(semantic_labels, semantic_labels)
    tdp_events = [make_event("{}", event_id="success"), make_event("{}", event_id="failed")]
    tdp = tdp_compatibility_metrics(
        tdp_events,
        ["confirmed_upload_success", "failed"],
        ["confirmed_upload_success", "failed"],
    )
    report = {
        "mode": "smoke",
        "run_id": "test",
        "all_gates_passed": True,
        "semantic_gold": semantic,
        "tdp_compatibility": tdp,
        "single_batch_parity": {
            "gold": {"mismatch_count": 0},
            "tdp": {"mismatch_count": 0},
        },
    }
    markdown = _acceptance_markdown(report)
    assert "TDP false-success" in markdown
    assert "non-3D" not in markdown
    assert "threat_level" not in markdown
