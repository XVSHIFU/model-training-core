from upload_judge.rules import apply_rules
from upload_judge.schemas import Verdict


def test_failure_overrides_success(make_event):
    result = apply_rules(make_event('{"success":false,"msg":"上传成功","url":"/uploads/x.php"}'))
    assert result and result.verdict == Verdict.FAILED


def test_strong_success_with_resource(make_event):
    result = apply_rules(make_event('{"success":true,"url":"/uploads/a.jpg"}', filename="a.jpg"))
    assert result and result.verdict == Verdict.CONFIRMED_UPLOAD_SUCCESS


def test_safe_extension_can_be_confirmed_with_direct_save_evidence(make_event):
    result = apply_rules(make_event('{"success":true,"fileId":"42"}', filename="part.step"))
    assert result and result.verdict == Verdict.CONFIRMED_UPLOAD_SUCCESS


def test_script_extension_not_directly_confirmed(make_event):
    assert apply_rules(make_event('{"success":true,"url":"/uploads/shell.php"}')) is None


def test_403_without_permission_semantics_is_not_automatic_failure(make_event):
    assert apply_rules(make_event('{"message":"business response"}', status=403)) is None


def test_login_page_is_failure(make_event):
    result = apply_rules(
        make_event(
            '<html><title>Login</title><form><input type="password"></form></html>',
            content_type="text/html",
        )
    )
    assert result and result.verdict == Verdict.FAILED


def test_login_words_inside_embedded_upload_html_are_not_failure(make_event):
    body = '<textarea>{success:true,value:"<html><script>var passwordPolicy=true;</script></html>"}</textarea>'
    assert apply_rules(make_event(body, filename="a.zip", content_type="text/html")) is None


def test_bare_ok_is_weak(make_event):
    assert apply_rules(make_event("OK", content_type="text/plain")) is None


def test_failure_code_overrides_success_true(make_event):
    result = apply_rules(make_event('{"success":true,"code":500,"message":"输入有误"}', filename="a.xlsx"))
    assert result and result.verdict == Verdict.FAILED


def test_v4_success_code_plus_filelink_is_deterministic_strong_success(make_event):
    result = apply_rules(make_event('{"code":200,"msg":"操作成功","filelink":"/files/a.pdf"}', filename="a.pdf"))
    assert result is not None
    assert result.verdict == Verdict.CONFIRMED_UPLOAD_SUCCESS


def test_explicit_upload_failure_text_is_a_strong_failure_rule(make_event):
    result = apply_rules(make_event('{"success":true,"message":"文件上传失败"}', filename="a.pdf"))
    assert result is not None
    assert result.verdict == Verdict.FAILED
