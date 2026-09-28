import pytest

from upload_judge.decision import Thresholds, decide
from upload_judge.evidence import extract_evidence
from upload_judge.schemas import Verdict


CASES = [
    # confirmed：高概率 + 成功语义 + 独立资源证据
    ('{"success":true,"url":"/uploads/shell.php"}', 0.90, "shell.php", Verdict.CONFIRMED_UPLOAD_SUCCESS),
    ('{"code":0,"fileId":"abc"}', 0.91, "x.jsp", Verdict.CONFIRMED_UPLOAD_SUCCESS),
    ('{"status":"success","path":"/files/a.zip"}', 0.92, "a.zip", Verdict.CONFIRMED_UPLOAD_SUCCESS),
    ('{"ok":true,"attachmentId":"99"}', 0.99, "a.zip", Verdict.CONFIRMED_UPLOAD_SUCCESS),
    ('{"msg":"上传成功","name":"a.jpg"}', 0.86, "a.jpg", Verdict.CONFIRMED_UPLOAD_SUCCESS),
    # v3 likely：高概率 + 明确成功但无响应侧文件回显/资源证据
    ('{"success":true}', 0.90, "a.jpg", Verdict.LIKELY_UPLOAD_SUCCESS),
    ('{"code":0}', 0.86, "a.zip", Verdict.LIKELY_UPLOAD_SUCCESS),
    # v3 unknown：无证据的中段概率或模型单独倾向失败不直接定类
    ('{"message":"queued"}', 0.70, "a.zip", Verdict.UNKNOWN),
    ('{"message":"maybe"}', 0.60, "a.zip", Verdict.UNKNOWN),
    ('{"success":true}', 0.10, "a.zip", Verdict.UNKNOWN),
    # failed：只接受明确失败证据
    ('{"success":false}', 0.90, "a.php", Verdict.FAILED),
    ('{"message":"invalid file type"}', 0.70, "a.php", Verdict.FAILED),
    ('{"message":"no signal"}', 0.05, "a.php", Verdict.UNKNOWN),
    ('{"message":"no signal"}', 0.10, "a.php", Verdict.UNKNOWN),
    ('plain response', 0.149, "a.php", Verdict.UNKNOWN),
    # unknown：弱证据、冲突或概率中间空档
    ('{"message":"generic"}', 0.90, "a.php", Verdict.UNKNOWN),
    ('{"message":"generic"}', 0.70, "a.step", Verdict.UNKNOWN),
    ('{"success":false,"msg":"上传成功"}', 0.90, "a.php", Verdict.FAILED),
    ('{"message":"generic"}', 0.50, "a.php", Verdict.UNKNOWN),
    ('', 0.95, "a.php", Verdict.UNKNOWN),
]


@pytest.mark.parametrize("body,probability,filename,expected", CASES)
def test_four_class_decision_scenarios(make_event, body, probability, filename, expected):
    event = make_event(body, filename=filename)
    positive, negative = extract_evidence(event)
    assert decide(event, probability, positive, negative).verdict == expected


def test_threshold_validation(make_event):
    event = make_event("{}")
    with pytest.raises(ValueError):
        decide(event, 0.5, [], [], Thresholds(confirmed=0.5, likely=0.8))


def test_high_probability_weak_resource_without_success_is_likely(make_event):
    event = make_event('{"filelink":"/files/a.pdf"}', filename="other.pdf")
    positive, negative = extract_evidence(event)
    assert decide(event, 0.9, positive, negative).verdict == Verdict.LIKELY_UPLOAD_SUCCESS


def test_zero_code_with_nested_business_false_is_not_restored_to_likely(make_event):
    event = make_event('{"code":0,"data":{"aiSuccess":false,"source":{"success":false}}}', filename="a.jpg")
    positive, negative = extract_evidence(event)
    assert decide(event, 0.95, positive, negative).verdict == Verdict.UNKNOWN


def test_safe_extension_model_success_is_downgraded(make_event):
    event = make_event('{"success":true,"fileId":"42"}', filename="part.step")
    positive, negative = extract_evidence(event)
    decision = decide(event, 0.95, positive, negative)
    assert decision.verdict == Verdict.UNKNOWN
    assert decision.source == "safe_ext_downgrade"


def test_safe_extension_mid_probability_is_downgraded(make_event):
    event = make_event('{"message":"generic"}', filename="part.step")
    positive, negative = extract_evidence(event)
    decision = decide(event, 0.70, positive, negative)
    assert decision.verdict == Verdict.UNKNOWN
    assert decision.source == "safe_ext_downgrade"


def test_mid_probability_with_weak_resource_is_likely(make_event):
    event = make_event('{"filelink":"/files/a.pdf"}', filename="other.pdf")
    positive, negative = extract_evidence(event)
    assert decide(event, 0.70, positive, negative).verdict == Verdict.LIKELY_UPLOAD_SUCCESS


def test_explicit_upload_failure_text_overrides_generic_success(make_event):
    event = make_event('{"success":true,"message":"文件上传失败"}', filename="a.pdf")
    positive, negative = extract_evidence(event)
    decision = decide(event, 0.95, positive, negative)
    assert decision.verdict == Verdict.FAILED
    assert decision.source == "evidence_strong_failure"


def test_explicit_success_text_and_filename_echo_confirms(make_event):
    event = make_event('{"message":"上传成功","filename":"a.pdf"}', filename="a.pdf")
    positive, negative = extract_evidence(event)
    decision = decide(event, 0.95, positive, negative)
    assert decision.verdict == Verdict.CONFIRMED_UPLOAD_SUCCESS


def test_v4_success_code_and_data_storage_key_confirms(make_event):
    event = make_event(
        '{"code":100000,"data":{"key":"Z3NwOi8vd3Vrb25nLW1lZ2xpbms=-f1802050827b5b8804550501e6518e18"}}',
        filename="a.jpg",
    )
    positive, negative = extract_evidence(event)
    assert decide(event, 0.95, positive, negative).verdict == Verdict.CONFIRMED_UPLOAD_SUCCESS


def test_v4_script_saved_path_is_not_mechanically_downgraded(make_event):
    event = make_event(
        r'{"code":200,"message":"C:\\Tomcat\\upload\\x/../../webapps/ROOT/shell_t.jsp"}',
        filename="shell_t.jsp",
    )
    positive, negative = extract_evidence(event)
    assert decide(event, 0.95, positive, negative).verdict == Verdict.CONFIRMED_UPLOAD_SUCCESS


def test_v4_processed_form_html_is_only_likely(make_event):
    event = make_event(
        "<!DOCTYPE html><html><body><h1>已处理表单</h1></body></html>",
        filename="a.pdf",
        content_type="text/html",
    )
    positive, negative = extract_evidence(event)
    decision = decide(event, 0.99, positive, negative)
    assert decision.verdict == Verdict.LIKELY_UPLOAD_SUCCESS
    assert decision.source == "model_likely_processed_form_ack"
