import pytest

from upload_judge.evidence import extract_evidence, has_failure_semantics, has_resource_evidence
from upload_judge.features import build_model_text, extract_structured_features, response_format


def test_extracts_success_path_and_filename(make_event):
    event = make_event('{"success":true,"url":"/uploads/shell.php"}')
    features = extract_structured_features(event)
    positive, _ = extract_evidence(event)
    assert "success=true" in features["success_fields"]
    assert "/uploads/shell.php" in features["paths"]
    assert features["filename_in_response"]
    assert has_resource_evidence(positive)


def test_model_text_truncates_large_body(make_event):
    event = make_event('{"success":true}' + "x" * 30000)
    text = build_model_text(event)
    assert len(text) < 8000
    assert "REQ_FILENAME=shell.php" in text


def test_response_formats(make_event):
    assert response_format(make_event("")) == "empty"
    assert response_format(make_event("{}")) == "json"
    assert response_format(make_event("<html></html>", content_type="text/html")) == "html"
    assert response_format(make_event("plain", content_type="text/plain")) == "text"


@pytest.mark.parametrize(
    "body,expected",
    [
        ('{"code":100000,"msg":"","data":{"key":"opaque"}}', "code=100000"),
        ('{"code":"100000"}', "code=100000"),
        ('{"IsSucess":true,"IsError":false,"Code":0}', "isSuccess=true"),
        ('{"isSuccess":true}', "isSuccess=true"),
        ('{"status":true,"data":{"tasktabuuid":"opaque"}}', "status=true"),
        ('{"retStatus":true}', "retStatus=true"),
        ('{"retCode":"0000"}', "retCode=0000"),
        ('{"retMsg":"成功"}', "retMsg=成功"),
    ],
)
def test_p05_success_semantics(make_event, body, expected):
    assert expected in extract_structured_features(make_event(body))["success_fields"]


def test_existing_result_success_pattern_is_not_duplicated(make_event):
    fields = extract_structured_features(make_event('{"result":"success"}'))["success_fields"]
    assert fields.count("state=success") == 1


@pytest.mark.parametrize("field", ["fileSaveName", "fileRealName"])
def test_p05_returned_filename_fields(make_event, field):
    body = f'{{"result":"success","{field}":"20260810-report.pdf"}}'
    features = extract_structured_features(make_event(body))
    assert "20260810-report.pdf" in features["returned_filenames"]


def test_p05_data_file_requires_extension(make_event):
    matched = extract_structured_features(make_event('{"data":[{"file":"report.pdf"}]}'))
    not_matched = extract_structured_features(make_event('{"data":[{"file":"opaque"}]}'))
    assert "report.pdf" in matched["returned_filenames"]
    assert not_matched["returned_filenames"] == []


def test_p05_obj_path_matches_direct_and_escaped_json(make_event):
    direct = extract_structured_features(make_event('{"obj":"upload/incomeAtt2/report.png"}'))
    escaped = extract_structured_features(
        make_event(r'{"jsonStr":"{\"obj\":\"upload/incomeAtt2/escaped.png\"}"}')
    )
    assert "upload/incomeAtt2/report.png" in direct["paths"]
    assert "upload/incomeAtt2/escaped.png" in escaped["paths"]


def test_p05_filename_echo_url_decodes_body_and_uri(make_event):
    body_event = make_event('{"name":"report%202026.pdf"}', filename="report 2026.pdf")
    uri_event = make_event("{}", filename="report 2026.pdf")
    uri_event.request.uri = "/upload?filename=report%202026.pdf"
    assert extract_structured_features(body_event)["filename_in_response"] is True
    assert extract_structured_features(uri_event)["filename_in_response"] is True


def test_p05_plain_request_uri_filename_is_not_server_echo(make_event):
    event = make_event("{}", filename="report.pdf")
    event.request.uri = "/upload?filename=report.pdf"
    assert extract_structured_features(event)["filename_in_response"] is False


def test_p05_nonterminal_status_suppresses_filename_resource(make_event):
    event = make_event(
        '{"filename":"report.pdf","status":"Uploading"}',
        filename="report.pdf",
    )
    features = extract_structured_features(event)
    positive, _ = extract_evidence(event)
    assert features["returned_filenames"] == []
    assert features["filename_in_response"] is False
    assert has_resource_evidence(positive) is False


def test_p05_terminal_status_keeps_filename_resource(make_event):
    event = make_event(
        '{"filename":"report.pdf","status":"success"}',
        filename="report.pdf",
    )
    features = extract_structured_features(event)
    assert features["returned_filenames"] == ["report.pdf"]
    assert features["filename_in_response"] is True


@pytest.mark.parametrize(
    "body",
    [
        '{"code":1}',
        '{"result":1}',
        '{"result":0}',
    ],
)
def test_p05_ambiguous_numeric_results_are_not_success(make_event, body):
    assert extract_structured_features(make_event(body))["success_fields"] == []


@pytest.mark.parametrize(
    "body",
    [
        '{"key":"opaque"}',
        '{"obj":"opaque"}',
        '{"obj":"report.pdf"}',
        'upload failed; see http://example.test/error/help',
    ],
)
def test_p05_ambiguous_resources_are_not_accepted(make_event, body):
    features = extract_structured_features(make_event(body))
    positive, _ = extract_evidence(make_event(body))
    assert features["paths"] == []
    assert features["file_ids"] == []
    assert features["returned_filenames"] == []
    assert has_resource_evidence(positive) is False


@pytest.mark.parametrize(
    "field",
    [
        "doc_id",
        "docId",
        "document_id",
        "copy_id",
        "copyId",
        "file_hash",
        "fileHash",
        "task_files_id",
        "taskFilesId",
        "DocumentGuid",
    ],
)
def test_p05b_resource_id_fields(make_event, field):
    body = f'{{"result":{{"files":[{{"{field}":"resource-123"}}]}}}}'
    features = extract_structured_features(make_event(body))
    assert "resource-123" in features["file_ids"]


def test_p05b_resource_id_matches_escaped_nested_json(make_event):
    body = r'{"jsonStr":"{\"result\":{\"files\":[{\"docId\":\"nested-123\"}]}}"}'
    features = extract_structured_features(make_event(body))
    assert "nested-123" in features["file_ids"]


@pytest.mark.parametrize(
    "resource_id",
    [
        "decf33bd-3306-43f1-a18e-b4a200d021c9",
        "6506bcfe37084c77a05ee617bafee320",
    ],
)
def test_p05b_result_guid_is_resource_id(make_event, resource_id):
    features = extract_structured_features(
        make_event(f'{{"result":"{resource_id}","success":true}}')
    )
    assert resource_id in features["file_ids"]


def test_p05b_artifact_is_resource_path(make_event):
    artifact = "https://files.example.test/multi_models/model.zip"
    features = extract_structured_features(
        make_event(f'{{"file_hash":"6506bcfe","artifact":"{artifact}"}}')
    )
    assert artifact in features["paths"]


@pytest.mark.parametrize(
    "body,expected",
    [
        ('<a href="/assets/report.pdf">附件</a>', "/assets/report.pdf"),
        ('<img src="/preview/image.png">', "/preview/image.png"),
        ("<script>location.href='/download/file?id=123'</script>", "/download/file?id=123"),
        ('<a href="https://files.example.test/attachments/123">附件</a>', "https://files.example.test/attachments/123"),
    ],
)
def test_p05b_html_resource_links(make_event, body, expected):
    features = extract_structured_features(make_event(body, content_type="text/html"))
    assert expected in features["paths"]


@pytest.mark.parametrize(
    "body",
    [
        '{"key":"opaque"}',
        '{"task_id":"task-123"}',
        '{"result":"success"}',
        '{"result":"ordinary-text"}',
        '{"commit":"decf33bd-3306-43f1-a18e-b4a200d021c9"}',
        '<a href="https://example.test/dashboard">首页</a>',
        '<img src="https://example.test/assets/logo">',
    ],
)
def test_p05b_ambiguous_values_are_not_resource_evidence(make_event, body):
    features = extract_structured_features(make_event(body, content_type="text/html"))
    positive, _ = extract_evidence(make_event(body, content_type="text/html"))
    assert features["paths"] == []
    assert features["file_ids"] == []
    assert features["returned_filenames"] == []
    assert has_resource_evidence(positive) is False


@pytest.mark.parametrize(
    "body",
    [
        '{"result":"decf33bd-3306-43f1-a18e-b4a200d021c9","success":true,"unAuthorizedRequest":false}',
        '{"success":true,"error":null}',
        '{"success":true,"error":""}',
        '{"success":true,"name":"未授权报告.pdf"}',
    ],
)
def test_p05c_failure_words_outside_nonempty_failure_values_do_not_match(make_event, body):
    _, negative = extract_evidence(make_event(body))
    assert has_failure_semantics(negative) is False


@pytest.mark.parametrize("container", ["sheetData", "allFileData"])
def test_p05c_row_level_import_failures_do_not_match(make_event, container):
    body = f'{{"success":true,"{container}":[{{"message":"文件上传失败","error":"校验失败"}}]}}'
    _, negative = extract_evidence(make_event(body))
    assert has_failure_semantics(negative) is False


def test_p05c_failure_message_value_matches(make_event):
    _, negative = extract_evidence(make_event('{"message":"文件上传失败"}'))
    assert has_failure_semantics(negative) is True


def test_p05c_403_forbidden_page_matches(make_event):
    body = "<html><head><title>403 Forbidden</title></head><body>Access Denied</body></html>"
    _, negative = extract_evidence(make_event(body, content_type="text/html"))
    assert has_failure_semantics(negative) is True


def test_p05c_top_level_success_false_still_matches(make_event):
    _, negative = extract_evidence(make_event('{"success":false}'))
    assert has_failure_semantics(negative) is True


@pytest.mark.parametrize(
    "body,filename,expected_score,expected_strength,expected_grade",
    [
        ('{"url":"/uploads/a.pdf","fileId":"42"}', "other.pdf", 2, "none", "STRONG"),
        ('{"success":true,"url":"/uploads/a.pdf"}', "other.pdf", 1, "explicit", "STRONG"),
        ('{"filelink":"/files/a.pdf"}', "other.pdf", 1, "none", "WEAK"),
        ('{"success":true}', "a.pdf", 0, "explicit", "WEAK"),
        ('{"msg":"操作成功"}', "a.pdf", 0, "generic", "NONE"),
    ],
)
def test_v3_evidence_grades(make_event, body, filename, expected_score, expected_strength, expected_grade):
    features = extract_structured_features(make_event(body, filename=filename))
    assert features["resource_score"] == expected_score
    assert features["success_strength"] == expected_strength
    assert features["evidence_grade"] == expected_grade


def test_v3_file_hash_is_its_own_resource_category(make_event):
    features = extract_structured_features(make_event('{"fileHash":"6506bcfe"}', filename="a.pdf"))
    assert features["file_hashes"] == ["6506bcfe"]
    assert features["resource_score"] == 1
    assert features["evidence_grade"] == "WEAK"


def test_v4_explicit_upload_failure_text_is_strong_despite_generic_success(make_event):
    features = extract_structured_features(
        make_event('{"success":true,"message":"文件上传失败"}', filename="a.pdf")
    )
    assert features["strong_failure"] is True
    assert features["weak_failure_conflict"] is False


def test_v3_deterministic_failure_remains_strong(make_event):
    features = extract_structured_features(
        make_event('{"success":true,"code":500,"message":"输入有误"}', filename="a.pdf")
    )
    assert features["strong_failure"] is True
    assert features["weak_failure_conflict"] is False


def test_v3_html_success_link_is_strong_evidence(make_event):
    event = make_event(
        '<html><body>上传成功 <a href="/uploads/a.pdf">a.pdf</a></body></html>',
        filename="a.pdf",
        content_type="text/html",
    )
    features = extract_structured_features(event)
    assert features["is_html"] is True
    assert features["success_strength"] == "explicit"
    assert features["evidence_grade"] == "STRONG"


def test_v3_html_resource_link_counts_toward_resource_score(make_event):
    features = extract_structured_features(
        make_event('<a href="/attachments/random.pdf">附件</a>', filename="other.pdf", content_type="text/html")
    )
    assert features["resource_score"] == 1
    assert features["evidence_grade"] == "WEAK"


def test_v3_invalid_json_success_and_fid_are_detected(make_event):
    features = extract_structured_features(
        make_event('{"success": true, "fid":"12345", "when":new Date()}', filename="a.pdf")
    )
    assert "success=true" in features["success_fields"]
    assert "12345" in features["resource_ids"]
    assert features["evidence_grade"] == "STRONG"


def test_v3_unicode_escaped_success_message_is_decoded(make_event):
    features = extract_structured_features(
        make_event(r'{"err_code":"0","err_msg":"\u6210\u529f","fid":"abc123"}', filename="a.pdf")
    )
    assert "code=0" in features["success_fields"]
    assert "success_message" in features["success_fields"]


def test_v3_preview_filename_is_not_saved_resource(make_event):
    event = make_event(
        '{"result":{"allFileData":[{"fileName":"preview.xlsx","sheetData":[]}]},"success":true}',
        filename="preview.xlsx",
    )
    event.request.uri = "/Excel/import/preview"
    features = extract_structured_features(event)
    assert features["returned_filenames"] == []
    assert features["filename_in_response"] is False


def test_v3_temp_paths_and_null_ids_are_not_resources(make_event):
    features = extract_structured_features(
        make_event('{"fileId":null,"absolutePath":"D:\\\\tmp\\\\upload_123.tmp"}', filename="a.jpg")
    )
    assert features["resource_ids"] == []
    assert features["paths"] == []


def test_v3_nested_success_false_is_failure_without_row_level_false_positive(make_event):
    nested = extract_structured_features(make_event('{"resphead":{"success":false}}'))
    row_level = extract_structured_features(make_event('{"sheetData":[{"success":false}]}'))
    assert "nested_success=false" in nested["fail_fields"]
    assert row_level["fail_fields"] == []


@pytest.mark.parametrize(
    "body,marker",
    [
        ('{"code":{"retcode":-6,"retinfo":"error"}}', "negative_retcode"),
        ('{"code":{"retcode":-6,"retinfo":"error"}}', "retinfo=error"),
        ('{"errorCode":610}', "nonzero_error_code"),
    ],
)
def test_v4_deterministic_error_code_patterns(make_event, body, marker):
    features = extract_structured_features(make_event(body))
    assert marker in features["fail_fields"]
    assert features["strong_failure"] is True


@pytest.mark.parametrize(
    "body",
    [
        '{"unAuthorizedRequest":false}',
        '{"error":null}',
        '{"aiSuccess":false}',
        '{"sheetData":[{"success":false,"message":"上传失败"}]}',
        '{"allFileData":[{"success":false,"message":"保存失败"}]}',
    ],
)
def test_v4_failure_patterns_do_not_harm_boundaries(make_event, body):
    features = extract_structured_features(make_event(body))
    assert features["fail_fields"] == []


def test_v4_status_false_is_failure_but_row_status_is_excluded(make_event):
    failed = extract_structured_features(make_event('{"code":"60404","status":false}'))
    row = extract_structured_features(make_event('{"sheetData":[{"status":false}]}'))
    assert "status=false" in failed["fail_fields"]
    assert failed["strong_failure"] is True
    assert row["fail_fields"] == []


def test_v4_nested_success_false_is_found_below_top_level_success(make_event):
    features = extract_structured_features(
        make_event('{"success":true,"data":{"success":false}}')
    )
    assert "success=true" in features["success_fields"]
    assert "nested_success=false" in features["fail_fields"]
    assert features["strong_failure"] is False


@pytest.mark.parametrize(
    "body",
    [
        "message=上传失败",
        "msg='保存失败'",
        "errorMessage=后缀不支持",
    ],
)
def test_v4_nonstandard_failure_message_fields(make_event, body):
    features = extract_structured_features(make_event(body, content_type="text/plain"))
    assert "upload_failure_text" in features["fail_fields"]
    assert features["strong_failure"] is True


@pytest.mark.parametrize(
    "body,marker",
    [
        ('{"status":"End"}', "state=complete"),
        ('{"state":"done"}', "state=complete"),
        ('{"status":"finished"}', "state=complete"),
        ('{"Success":1}', "success=1"),
        ('Success=true', "success=1"),
        ('{"state":true}', "state=true"),
    ],
)
def test_v4_success_semantics(make_event, body, marker):
    assert marker in extract_structured_features(make_event(body))["success_fields"]


def test_v4_ret_zero_and_fmcode_resource_are_success_evidence(make_event):
    ret = extract_structured_features(make_event('{"ret":0,"url":"/files/a.pdf"}', filename="a.pdf"))
    fmcode = extract_structured_features(make_event('{"fmcode":0,"data":[1897219]}', filename="a.jpg"))
    assert "code=0" in ret["success_fields"]
    assert ret["evidence_grade"] == "STRONG"
    assert "code=0" in fmcode["success_fields"]
    assert "1897219" in fmcode["file_ids"]
    assert fmcode["evidence_grade"] == "STRONG"


def test_v4_dynamic_upload_status_is_explicit_success(make_event):
    features = extract_structured_features(
        make_event('{"UPLOAD_STATUS":{"a.zip":"SUCCESS"},"STATUS_CODE":"0"}', filename="a.zip")
    )
    assert "upload_status=success" in features["success_fields"]
    assert features["evidence_grade"] == "STRONG"


@pytest.mark.parametrize("body", ["Upload", "1", '"1"'])
def test_v4_scalar_ack_only_counts_in_upload_context(make_event, body):
    upload = make_event(body, content_type="text/plain", filename="a.pdf")
    other = make_event(body, content_type="text/plain", filename="a.pdf")
    other.request.uri = "/api/process"
    assert extract_structured_features(upload)["success_strength"] == "explicit"
    assert extract_structured_features(other)["success_fields"] == []


def test_v4_success_code_and_data_storage_key_is_strong_resource(make_event):
    key = "Z3NwOi8vd3Vrb25nLW1lZ2xpbms=-f1802050827b5b8804550501e6518e18"
    features = extract_structured_features(
        make_event(f'{{"code":100000,"data":{{"key":"{key}"}}}}', filename="a.jpg")
    )
    assert features["storage_keys"] == [key]
    assert key in features["file_ids"]
    assert features["evidence_grade"] == "STRONG"


def test_v4_opaque_data_key_is_not_strong_resource(make_event):
    features = extract_structured_features(
        make_event('{"code":100000,"data":{"key":"opaque"}}', filename="a.jpg")
    )
    assert features["storage_keys"] == ["opaque"]
    assert "opaque" not in features["file_ids"]
    assert features["evidence_grade"] == "WEAK"


def test_v4_base64_business_key_without_decoded_path_stays_weak(make_event):
    key = "1_ZzEwMF82bQ==_29fe33ed9585459580bd9c0c3a5a0a1c"
    features = extract_structured_features(
        make_event(f'{{"code":100000,"data":{{"key":"{key}"}}}}', filename="a.jpg")
    )
    assert key not in features["file_ids"]
    assert features["evidence_grade"] == "WEAK"


def test_v4_message_saved_path_is_resource(make_event):
    path = r"C:\\Tomcat\\upload\\x/../../webapps/ROOT/shell_t.jsp"
    features = extract_structured_features(
        make_event(f'{{"code":200,"message":"{path}"}}', filename="shell_t.jsp")
    )
    assert path in features["paths"]
    assert features["evidence_grade"] == "STRONG"
