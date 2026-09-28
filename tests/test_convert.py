import gzip

import brotli

from upload_judge.convert import convert_http_dump, convert_record, parse_headers


def test_sensitive_headers_redacted():
    headers = parse_headers("Authorization: secret\r\nCookie: a=b\r\nContent-Type: text/plain")
    assert headers["Authorization"] == "<redacted>"
    assert headers["Cookie"] == "<redacted>"
    assert headers["Content-Type"] == "text/plain"


def test_convert_tdp_record_has_stable_fallback_id():
    raw = {
        "detail": {"data": {"detail": {"alert": {"net": {"http": {
            "method": "POST",
            "url": "https://example.test/upload?q=1",
            "reqs_header": "Content-Type: multipart/form-data\r\nAuthorization: secret",
            "reqs_body": 'Content-Disposition: form-data; name="file"; filename="shell.php"',
            "resp_header": "Content-Type: application/json",
            "resp_body": '{"success":true}',
            "status": "200",
        }}}}}}
    }
    item = convert_record(raw, "confirmed_upload_success", 7)
    assert item is not None
    assert item["event_id"] == "tdp_confirmed_upload_success_7"
    assert item["request"]["uri"] == "/upload?q=1"
    assert item["request"]["filename"] == "shell.php"
    assert item["request"]["headers"]["Authorization"] == "<redacted>"


def test_convert_http_dump_with_blank_lines():
    raw = (
        "POST /api/upload HTTP/1.1\n"
        "Host: example.test\n"
        "Authorization: secret\n"
        "Content-Type: multipart/form-data; boundary=----x\n"
        "\n"
        "------x\n"
        'Content-Disposition: form-data; name="file"; filename="shell.php"\n'
        "------x\n"
        "HTTP/1.1 200 OK\n"
        "Content-Type: application/json\n"
        "\n"
        '{"success":true,"data":{"url":"/u/1.php"}}'
    )
    item = convert_http_dump(raw)
    assert item["request"]["method"] == "POST"
    assert item["request"]["uri"] == "/api/upload"
    assert item["request"]["filename"] == "shell.php"
    assert item["request"]["field_name"] == "file"
    assert item["request"]["headers"]["Authorization"] == "<redacted>"
    assert item["response"]["status_code"] == 200
    assert item["response"]["body"] == '{"success":true,"data":{"url":"/u/1.php"}}'


def test_convert_http_dump_no_blank_line_before_body():
    # 部分抓包导出省略头/体之间的空行，multipart 边界直接跟在头后
    raw = (
        "POST /etap/files/import HTTP/1.0\n"
        "Host: a.b:443\n"
        "Content-Type: multipart/form-data; boundary=----WebKitBoundary\n"
        "------WebKitBoundary\n"
        'Content-Disposition: form-data; name="files"; filename="a.xlsx"\n'
        "Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet\n"
        "PKbinary-content\n"
        "HTTP/1.1 200 OK\n"
        "Content-Type: application/json\n"
        '{"success":false,"code":500,"message":"数据异常"}'
    )
    item = convert_http_dump(raw)
    assert item["request"]["content_type"].startswith("multipart/form-data")
    assert item["request"]["filename"] == "a.xlsx"
    assert item["request"]["file_ext"] == "xlsx"
    assert item["request"]["field_name"] == "files"
    assert item["response"]["status_code"] == 200
    assert "success" in item["response"]["body"]
    # 头不应被 multipart 内部 Content-Type 污染
    assert item["request"]["headers"].get("Content-Type", "").startswith("multipart")


def test_convert_http_dump_recover_mojibake_filename():
    # 含替换符（已丢失字节）的 GBK 双解码文件名做 best-effort 恢复
    raw = (
        "POST /u HTTP/1.1\n"
        "Content-Type: multipart/form-data; boundary=--b\n"
        "\n"
        "--b\n"
        'Content-Disposition: form-data; name="f"; filename="鐧嬎鸁电力交易�.xlsx"\n'
        "--b\n"
    )
    item = convert_http_dump(raw)
    assert "xlsx" in item["request"]["filename"]
    # 恢复后应保留关键业务词（电力交易）
    assert "电力交易" in item["request"]["filename"]


def _compressed_http_dump(encoding, payload):
    request = (
        b"POST /api/upload HTTP/1.1\r\n"
        b"Content-Type: multipart/form-data; boundary=--x\r\n\r\n"
        b"--x\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.docx\"\r\n\r\n"
    )
    response = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json; charset=utf-8\r\n"
        + f"Content-Encoding: {encoding}\r\n\r\n".encode("ascii")
        + payload
    )
    return request + response


def test_convert_http_dump_decompresses_gzip_from_original_bytes():
    body = '{"success":true,"message":"上传成功"}'.encode("utf-8")
    item = convert_http_dump(_compressed_http_dump("gzip", gzip.compress(body)))
    assert item["response"]["body"] == body.decode("utf-8")


def test_convert_http_dump_decompresses_brotli_from_original_bytes():
    body = '{"success":true,"message":"保存成功"}'.encode("utf-8")
    item = convert_http_dump(_compressed_http_dump("br", brotli.compress(body)))
    assert item["response"]["body"] == body.decode("utf-8")
