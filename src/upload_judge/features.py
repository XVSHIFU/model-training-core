"""文本标准化及规则/模型共享的结构化特征。"""

from __future__ import annotations

import base64
import json
import re
from html import unescape as html_unescape
from typing import Any
from urllib.parse import unquote

from upload_judge.schemas import UploadEvent

_PATH_PATTERNS = [
    re.compile(r'["\'](?:url|file_?url|download_?url|http_?url|img_?url|path|file_?path|absolute_?path|save_?path|object_?key|file_?link|load_?link|location|src|href|artifact)["\']\s*:\s*["\']([^"\']+)["\']', re.I),
    re.compile(r'["\']obj["\']\s*:\s*["\']([^"\']*[\\/][^"\']+)["\']', re.I),
    re.compile(r"/(?:upload|uploads|files|attachment|attachments|media|static)/[^\s\"'<>]+", re.I),
]
_MESSAGE_VALUE_PATTERNS = [
    re.compile(r'["\'](?:message|msg)["\']\s*:\s*["\']([^"\']+)["\']', re.I),
    re.compile(r'(?:^|[,;{\s])(?:message|msg)\s*=\s*["\']?([^"\'\r\n,;}]+)', re.I),
]
_ID_PATTERNS = [
    re.compile(r'["\'](?:file_?id|file_?key|file_?id_?code|fid|attachment_?id|attach_?id|media_?id|resource_?id|imagefileid|oss_?id|ref_?id|note_?sn|file_?system_?uuid|upload_?file_?access_?id|file_?info_?access_?id|doc_?id|document_?id|document_?guid|copy_?id|file_?hash|task_?files_?id)["\']\s*:\s*["\']?([^"\',}\s]+)', re.I),
    re.compile(r'["\']result["\']\s*:\s*["\']((?:[a-f0-9]{32}|[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}))["\']', re.I),
]
_RESOURCE_ID_PATTERNS = [
    re.compile(r'["\'](?:file_?id|file_?key|file_?id_?code|fid|attachment_?id|attach_?id|media_?id|resource_?id|imagefileid|oss_?id|ref_?id|note_?sn|file_?system_?uuid|upload_?file_?access_?id|file_?info_?access_?id|doc_?id|document_?id|document_?guid|copy_?id|task_?files_?id)["\']\s*:\s*["\']?([^"\',}\s]+)', re.I),
    re.compile(r'["\']result["\']\s*:\s*["\']((?:[a-f0-9]{32}|[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}))["\']', re.I),
]
_FILE_HASH_PATTERNS = [
    re.compile(r'["\']file_?hash["\']\s*:\s*["\']?([^"\',}\s]+)', re.I),
]
_HTML_LINK_PATTERNS = [
    re.compile(r'<a\b[^>]*?\bhref\s*=\s*["\']([^"\']+)["\']', re.I),
    re.compile(r'<img\b[^>]*?\bsrc\s*=\s*["\']([^"\']+)["\']', re.I),
    re.compile(r'\blocation\.href\s*=\s*["\']([^"\']+)["\']', re.I),
]
_HTML_RESOURCE_EXTENSION_PATTERN = re.compile(
    r'\.(?:pdf|jpe?g|png|gif|webp|bmp|tiff?|svg|xlsx?|xlsm|docx?|docm|pptx?|pptm|zip|rar|7z|tar|gz|tgz|csv|txt|rtf|odt|ods|odp|xml|json|mp3|wav|mp4|avi|mov|webm|php[3457]?|phtml|phar|jspx?|jspf|asp|aspx|ashx|war|jar|exe|dll|sh|bat|cmd|ps1|cgi|pl|step|stp|stl|obj|dxf|dwg|igs|iges|glb|gltf|fbx|skp|sldprt|sldasm)$',
    re.I,
)
_HTML_RESOURCE_SEGMENT_PATTERN = re.compile(
    r'(?:^|[\\/])(?:uploads?|attachments?|downloads?|files)(?:[\\/?#]|$)',
    re.I,
)
_RETURNED_FILENAME_PATTERNS = [
    re.compile(r'["\'](?:file_?name|filename|full_?name|original_?name|original_?file_?name|old_?name|object_?name|file_?save_?name|file_?real_?name)["\']\s*:\s*["\']([^"\']+)["\']', re.I),
    re.compile(r'["\']file["\']\s*:\s*["\']([^"\']+\.[a-z0-9]{1,16})["\']', re.I),
]
_SUCCESS_PATTERNS = [
    (re.compile(r'(?:["\']success["\']|[{,]\s*success)\s*:\s*true\b', re.I), "success=true"),
    (re.compile(r'(?:^|[{,;\s])["\']?success["\']?\s*(?:=\s*(?:true|1)\b|:\s*["\']?1["\']?\b)', re.I), "success=1"),
    (re.compile(r'["\']code["\']\s*:\s*(?:["\']100000["\']|100000)(?=\s*[,}\]])', re.I), "code=100000"),
    (re.compile(r'["\'](?:code|err_?code|errno|error_code|ret_?code|return_?code|status)["\']\s*:\s*["\']?0+["\']?\b', re.I), "code=0"),
    (re.compile(r'["\'](?:ret|fmcode)["\']\s*:\s*["\']?0+["\']?\b', re.I), "code=0"),
    (re.compile(r'["\'](?:code|status|status_?code)["\']\s*:\s*["\']?20[01]["\']?\b', re.I), "code=2xx"),
    (re.compile(r'["\'](?:status|state|result|upload_?status)["\']\s*:\s*["\'](?:success|ok|uploaded|created)["\']', re.I), "state=success"),
    (re.compile(r'(?:^|[{,;\s])["\']?(?:status|state)["\']?\s*[:=]\s*["\']?(?:end|done|complete|completed|finished)["\']?\b', re.I), "state=complete"),
    (re.compile(r'(?:^|[{,;\s])["\']?state["\']?\s*[:=]\s*(?:true|["\']true["\'])\b', re.I), "state=true"),
    (re.compile(r'["\']ok["\']\s*:\s*true\b', re.I), "ok=true"),
    (re.compile(r'["\'](?:issucess|issuccess)["\']\s*:\s*true\b', re.I), "isSuccess=true"),
    (re.compile(r'["\']status["\']\s*:\s*true\b', re.I), "status=true"),
    (re.compile(r'["\']retstatus["\']\s*:\s*true\b', re.I), "retStatus=true"),
    (re.compile(r'["\']retcode["\']\s*:\s*["\']0000["\']', re.I), "retCode=0000"),
    (re.compile(r'["\']retmsg["\']\s*:\s*["\']成功["\']', re.I), "retMsg=成功"),
    (re.compile(r'["\'](?:msg|message|note|err_?msg|result_?str)["\']\s*:\s*["\'](?:success|successful|成功|操作成功|请求成功|上传成功|保存成功)["\']', re.I), "success_message"),
    (re.compile(r"上传成功|保存成功|文件已保存|附件添加成功|附件已绑定|资源创建完成|素材已入库|uploaded\s+successfully|upload\s+success(?:ful)?|saved\s+successfully", re.I), "upload_success_text"),
]
_FAIL_PATTERNS = [
    (re.compile(r'["\'](?:code|status)["\']\s*:\s*["\']?(?:4\d\d|5\d\d\d?)["\']?\b', re.I), "failure_code"),
    (re.compile(r'["\']retcode["\']\s*:\s*["\']?-[1-9]\d*["\']?\b', re.I), "negative_retcode"),
    (re.compile(r'["\']retinfo["\']\s*:\s*["\']?error["\']?\b', re.I), "retinfo=error"),
    (re.compile(r'["\']error_?code["\']\s*:\s*["\']?[1-9]\d*["\']?\b', re.I), "nonzero_error_code"),
    (re.compile(r'["\'](?:state|status)["\']\s*:\s*["\'](?:error|failed|failure)["\']', re.I), "state=error"),
    (re.compile(r'["\']ok["\']\s*:\s*false\b', re.I), "ok=false"),
]
_FAILURE_VALUE_FIELDS = {
    "msg", "message", "error", "errmsg", "errormessage", "resultmsg",
    "exception", "description", "code", "status", "errcode", "errorcode",
    "state", "result",
}
_ROW_LEVEL_RESULT_FIELDS = {"sheetdata", "allfiledata"}
_FAILURE_VALUE_PATTERN = re.compile(
    r"上传失败|保存失败|解析文件失败|导入失败|(?:文件)?类型不允许|后缀不支持|不允许的文件|权限不足|未授权|校验失败|"
    r"forbidden|access denied|permission denied|unauthorized|login required|invalid file type|extension not allowed|file too large|blocked by policy|"
    r"(?:waf|virus|malware).{0,30}(?:blocked|detected)|(?:blocked|rejected).{0,30}(?:waf|virus|malware)",
    re.I,
)
_EXPLICIT_UPLOAD_FAILURE_VALUE_PATTERN = re.compile(
    r"上传失败|保存失败|(?:文件)?类型不允许|后缀不支持|不允许的文件|"
    r"upload\s+fail(?:ed|ure)?|save\s+fail(?:ed|ure)?|invalid file type|extension not allowed",
    re.I,
)
_JSON_LIKE_FAILURE_FIELD_PATTERN = re.compile(
    r'["\'](?:msg|message|error|err_?msg|error_?message|result_?msg|exception|description|code|status|errcode|error_code|state|result)["\']\s*:\s*["\']([^"\']+)["\']',
    re.I,
)
_NONSTANDARD_FAILURE_FIELD_PATTERN = re.compile(
    r'(?:^|[,;{\s])["\']?(?:msg|message|error_?message)["\']?\s*=\s*["\']?([^"\'\r\n,;}]+)',
    re.I,
)
_HTML_ERROR_CONTEXT_PATTERN = re.compile(
    r'<(?:title|h[1-6])\b[^>]*>(.*?)</(?:title|h[1-6])\s*>',
    re.I | re.S,
)
_EXCEPTION_PAGE_PATTERN = re.compile(
    r"traceback\s*\(most recent call last\)|(?:^|\b)(?:[a-z_][\w.]*exception)(?::|\b)|\b403\s+forbidden\b|\baccess denied\b",
    re.I,
)
_NONTERMINAL_STATUS_PATTERN = re.compile(
    r'["\']status["\']\s*:\s*["\'](?:uploading|pending|processing)["\']',
    re.I,
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def build_model_text(event: UploadEvent) -> str:
    req, resp = event.request, event.response
    location = _get_header(resp.headers, "location")
    return "\n".join(
        [
            f"REQ_METHOD={req.method.upper()}",
            f"REQ_URI={req.uri[:1000]}",
            f"REQ_FILENAME={req.filename[:300]}",
            f"REQ_EXT={req.file_ext.lower().strip('.')}",
            f"REQ_FIELD={req.field_name[:200]}",
            f"REQ_BODY={normalize_text(req.body_excerpt)[:500]}",
            f"RESP_STATUS={resp.status_code}",
            f"RESP_CT={resp.content_type[:300]}",
            f"RESP_LOCATION={location[:1000]}",
            f"RESP_BODY={normalize_text(resp.body)[:5000]}",
        ]
    )


def extract_structured_features(event: UploadEvent) -> dict[str, Any]:
    req, resp = event.request, event.response
    body = (resp.body or "")[:20000]
    lower = body.lower()
    location = _get_header(resp.headers, "location")
    resource_body = _unescape_json_quotes(body)
    body_paths = [value for value in _unique_matches(resource_body, _PATH_PATTERNS) if _is_valid_resource_path(value)]
    for message in _unique_matches(resource_body, _MESSAGE_VALUE_PATTERNS):
        if _is_saved_path_message(message) and message not in body_paths:
            body_paths.append(message)
    paths = list(body_paths)
    for html_path in _unique_matches(resource_body, _HTML_LINK_PATTERNS):
        if _is_html_resource_link(html_path) and html_path not in paths:
            paths.append(html_path)
            body_paths.append(html_path)
    if location and location not in paths:
        paths.insert(0, location)
    file_ids = _unique_matches(resource_body, _ID_PATTERNS)
    resource_ids = _unique_matches(resource_body, _RESOURCE_ID_PATTERNS)
    file_hashes = _unique_matches(resource_body, _FILE_HASH_PATTERNS)
    returned_filenames = _unique_matches(resource_body, _RETURNED_FILENAME_PATTERNS)
    nonterminal_status = bool(_NONTERMINAL_STATUS_PATTERN.search(body))
    if nonterminal_status:
        returned_filenames = []
    filename = req.filename or ""
    filename_context = bool(filename.strip())
    basename = filename.rsplit(".", 1)[0] if filename else ""
    decoded_body = unquote(body).lower()
    raw_uri = req.uri or ""
    decoded_uri = unquote(raw_uri)
    # URI 是请求侧字段：只接受确实经过 %XX 解码后才出现的文件名，避免把普通
    # filename 查询参数误当成服务端回显。响应正文则始终允许解码后匹配。
    decoded_uri_for_echo = decoded_uri.lower() if decoded_uri != raw_uri else ""
    filename_in_response = bool(
        filename
        and (
            filename.lower() in decoded_body
            or filename.lower() in decoded_uri_for_echo
        )
    )
    if not filename_in_response and len(basename) >= 3:
        filename_in_response = (
            basename.lower() in decoded_body
            or basename.lower() in decoded_uri_for_echo
        )
    if nonterminal_status:
        filename_in_response = False
    preview_container = bool(
        re.search(r"(?:^|[/?#])preview(?:[/?#]|$)", req.uri or "", re.I)
        and re.search(r'["\'](?:sheetData|allFileData)["\']\s*:', body, re.I)
    )
    if preview_container:
        returned_filenames = []
        filename_in_response = False
    semantic_body = _decode_unicode_escapes(body)
    success_fields = [name for pattern, name in _SUCCESS_PATTERNS if pattern.search(semantic_body)]
    fail_fields = [name for pattern, name in _FAIL_PATTERNS if pattern.search(body)]
    parsed_body = _parse_json_value(body)
    upload_endpoint_context = bool(
        filename_context
        and re.search(r"(?:upload|attachment|file)", req.uri or "", re.I)
    )
    scalar_body = body.strip().strip('"\'').lower()
    if upload_endpoint_context and scalar_body == "upload":
        success_fields.append("upload_ack")
    if upload_endpoint_context and scalar_body == "1":
        success_fields.append("scalar_upload_ack")
    content_type = resp.content_type or _get_header(resp.headers, "content-type")
    content_type_is_json = "json" in content_type.lower()
    if parsed_body is not None:
        if _has_upload_status_success(parsed_body):
            success_fields.append("upload_status=success")
        if _has_failure_field_value(parsed_body):
            fail_fields.append("failure_text")
        if _has_explicit_upload_failure_value(parsed_body):
            fail_fields.append("upload_failure_text")
        if _has_false_field(parsed_body, "status"):
            fail_fields.append("status=false")
    elif content_type_is_json:
        if _json_like_failure_field_value(body):
            fail_fields.append("failure_text")
        if _json_like_explicit_upload_failure_value(body):
            fail_fields.append("upload_failure_text")
    elif _has_non_json_failure_page(body):
        fail_fields.append("failure_page")
    if parsed_body is None:
        nonstandard_failure = _nonstandard_failure_field_value(body)
        if nonstandard_failure:
            fail_fields.append("failure_text")
            if _EXPLICIT_UPLOAD_FAILURE_VALUE_PATTERN.search(nonstandard_failure):
                fail_fields.append("upload_failure_text")
        if re.search(r'["\']status["\']\s*:\s*false\b', body, re.I):
            fail_fields.append("status=false")
    top_level = parsed_body if isinstance(parsed_body, dict) else {}
    if top_level.get("success") is True:
        if "success=true" not in success_fields:
            success_fields.insert(0, "success=true")
    elif top_level.get("success") is False:
        fail_fields.insert(0, "success=false")
    if parsed_body is not None and _has_descendant_success_false(parsed_body):
        fail_fields.insert(0, "nested_success=false")
    elif parsed_body is None and re.search(r'["\']success["\']\s*:\s*false\b', body, re.I):
        fail_fields.insert(0, "nested_success=false")
    storage_keys = _data_storage_keys(parsed_body, resource_body)
    strong_storage_keys = [value for value in storage_keys if _looks_like_storage_key(value)]
    if any(value in success_fields for value in ("code=0", "code=100000", "code=2xx")):
        for value in strong_storage_keys:
            if value not in file_ids:
                file_ids.append(value)
            if value not in resource_ids:
                resource_ids.append(value)
    for value in _fmcode_resource_ids(parsed_body):
        if value not in file_ids:
            file_ids.append(value)
        if value not in resource_ids:
            resource_ids.append(value)
    is_json = content_type_is_json or parsed_body is not None
    is_html = "html" in content_type.lower() or bool(re.search(r"<(?:!doctype\s+html|html|body|form)\b", lower))
    success_strength = _success_strength(success_fields, filename_context)
    resource_score = sum(
        (
            bool(body_paths),
            bool(resource_ids),
            bool(location),
            bool(returned_filenames or filename_in_response),
            bool(file_hashes),
        )
    )
    evidence_grade = _evidence_grade(
        resource_score=resource_score,
        success_strength=success_strength,
        status_code=resp.status_code,
    )
    strong_failure = any(
        value in {
            "success=false", "failure_code", "negative_retcode", "retinfo=error",
            "nonzero_error_code", "state=error", "ok=false", "failure_page",
            "upload_failure_text", "status=false",
        }
        for value in fail_fields
    )
    weak_failure_conflict = bool(success_fields and fail_fields and not strong_failure)
    return {
        "status_code": resp.status_code,
        "body": body,
        "body_lower": lower,
        "body_empty": not body.strip(),
        "content_type": content_type,
        "is_json": is_json,
        "is_html": is_html,
        "paths": paths,
        "file_ids": file_ids,
        "resource_ids": resource_ids,
        "file_hashes": file_hashes,
        "storage_keys": storage_keys,
        "returned_filenames": returned_filenames,
        "location": location,
        "filename": filename,
        "file_ext": (req.file_ext or "").lower().strip("."),
        "filename_in_response": filename_in_response,
        "success_fields": success_fields,
        "fail_fields": fail_fields,
        "resource_score": resource_score,
        "success_strength": success_strength,
        "evidence_grade": evidence_grade,
        "filename_context": filename_context,
        "top_level_success_true": top_level.get("success") is True,
        "strong_failure": strong_failure,
        "weak_failure_conflict": weak_failure_conflict,
    }


def _success_strength(success_fields: list[str], filename_context: bool) -> str:
    values = set(success_fields)
    if "upload_success_text" in values:
        return "explicit"
    upload_context_success = {
        "success=true",
        "success=1",
        "code=0",
        "code=100000",
        "code=2xx",
        "state=success",
        "state=complete",
        "state=true",
        "ok=true",
        "isSuccess=true",
        "status=true",
        "retStatus=true",
        "retCode=0000",
        "retMsg=成功",
        "upload_ack",
        "scalar_upload_ack",
        "upload_status=success",
    }
    if filename_context and values & upload_context_success:
        return "explicit"
    if values:
        return "generic"
    return "none"


def _evidence_grade(resource_score: int, success_strength: str, status_code: int) -> str:
    if resource_score >= 2 or (resource_score >= 1 and success_strength == "explicit"):
        return "STRONG"
    if resource_score >= 1 or (
        resource_score == 0 and success_strength == "explicit" and 200 <= status_code < 300
    ):
        return "WEAK"
    return "NONE"


def response_format(event: UploadEvent) -> str:
    features = extract_structured_features(event)
    if features["body_empty"]:
        return "empty"
    if features["is_json"]:
        return "json"
    if features["is_html"]:
        return "html"
    return "text"


def _get_header(headers: dict[str, str], name: str) -> str:
    target = name.lower()
    return next((str(value) for key, value in (headers or {}).items() if key.lower() == target), "")


def _unescape_json_quotes(text: str) -> str:
    """仅为字段模式识别展开嵌套 JSON 的转义引号，不改变原始特征文本。"""
    return (text or "").replace(r'\"', '"').replace(r"\'", "'")


def _unique_matches(body: str, patterns: list[re.Pattern[str]]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(body):
            value = (match.group(1) if match.lastindex else match.group(0)).strip()
            if value and value.lower() not in {"null", "none", "undefined", "false", "true"} and value not in seen:
                seen.add(value)
                values.append(value)
    return values


def _is_html_resource_link(value: str) -> bool:
    candidate = (value or "").strip()
    return bool(
        _HTML_RESOURCE_EXTENSION_PATTERN.search(candidate)
        or _HTML_RESOURCE_SEGMENT_PATTERN.search(candidate)
    )


def _is_valid_resource_path(value: str) -> bool:
    candidate = (value or "").strip()
    if not candidate or re.fullmatch(r"[a-z]:[\\/]*", candidate, re.I):
        return False
    normalized = candidate.replace("\\\\", "/").replace("\\", "/").lower()
    if re.search(r"(?:^|/)(?:tmp|temp)(?:/|$)", normalized):
        return False
    return True


def _is_saved_path_message(value: str) -> bool:
    """只把 message/msg 中形似服务端保存位置的值作为资源路径。"""
    candidate = (value or "").strip()
    normalized = candidate.replace("\\\\", "/").replace("\\", "/")
    return bool(
        "/" in normalized
        and re.search(r"\.[a-z0-9]{1,16}(?:$|[?#\s])", normalized, re.I)
        and re.search(r"(?:^|/)(?:webapps?|uploads?|files?|attachments?)(?:/|$)", normalized, re.I)
        and _is_valid_resource_path(candidate)
    )


def _looks_like_storage_key(value: str) -> bool:
    candidate = (value or "").strip()
    if len(candidate) < 16 or candidate.lower() in {"null", "none", "undefined", "false", "true"}:
        return False
    if "/" in candidate or "\\" in candidate or re.search(r"\.[a-z0-9]{1,16}(?:$|[?#])", candidate, re.I):
        return True
    for token in re.split(r"[-_]", candidate):
        if len(token) < 12 or not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", token):
            continue
        try:
            decoded = base64.b64decode(token + "=" * (-len(token) % 4), validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if "://" in decoded or "/" in decoded or "\\" in decoded:
            return True
    return False


def _data_storage_keys(parsed_body: Any | None, resource_body: str) -> list[str]:
    """提取精确的 ``data.key``，避免把任意业务 ``key`` 当作文件资源。"""
    values: list[str] = []
    data = parsed_body.get("data") if isinstance(parsed_body, dict) else None
    containers = data if isinstance(data, list) else [data]
    for container in containers:
        if isinstance(container, dict):
            value = container.get("key")
            if isinstance(value, (str, int)):
                candidate = str(value).strip()
                if candidate and candidate.lower() not in {"null", "none", "undefined", "false", "true"}:
                    values.append(candidate)
    if parsed_body is None:
        pattern = re.compile(
            r'["\']data["\']\s*:\s*\{[^{}]{0,2000}?["\']key["\']\s*:\s*["\']([^"\']+)["\']',
            re.I | re.S,
        )
        values.extend(match.group(1).strip() for match in pattern.finditer(resource_body or ""))
    return list(dict.fromkeys(value for value in values if value))


def _fmcode_resource_ids(parsed_body: Any | None) -> list[str]:
    if not isinstance(parsed_body, dict) or parsed_body.get("fmcode") not in {0, "0"}:
        return []
    data = parsed_body.get("data")
    values = data if isinstance(data, list) else [data]
    return [
        str(value)
        for value in values
        if isinstance(value, (int, str))
        and not isinstance(value, bool)
        and str(value).strip()
        and str(value).strip() not in {"0", "null", "None"}
    ]


def _has_upload_status_success(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    for key, item in value.items():
        if _normalized_field_name(str(key)) != "uploadstatus":
            continue
        if isinstance(item, str) and item.strip().lower() in {"success", "successful", "uploaded"}:
            return True
        if isinstance(item, dict) and any(
            isinstance(status, str) and status.strip().lower() in {"success", "successful", "uploaded"}
            for status in item.values()
        ):
            return True
    return False


def _decode_unicode_escapes(value: str) -> str:
    return re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda match: chr(int(match.group(1), 16)),
        value or "",
    )


def _normalized_field_name(value: str) -> str:
    return re.sub(r"[_-]", "", (value or "").lower())


def _has_failure_field_value(value: Any, *, excluded: bool = False) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = _normalized_field_name(str(key))
            child_excluded = excluded or normalized in _ROW_LEVEL_RESULT_FIELDS
            if child_excluded:
                continue
            if normalized in _FAILURE_VALUE_FIELDS and isinstance(item, str):
                text = item.strip()
                if text and _FAILURE_VALUE_PATTERN.search(text):
                    return True
            if _has_failure_field_value(item, excluded=child_excluded):
                return True
        return False
    if isinstance(value, list):
        return any(_has_failure_field_value(item, excluded=excluded) for item in value)
    if isinstance(value, str) and not excluded:
        embedded = _parse_json_value(value)
        return embedded is not None and _has_failure_field_value(embedded)
    return False


def _has_nested_success_false(value: Any, *, excluded: bool = False) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = _normalized_field_name(str(key))
            child_excluded = excluded or normalized in _ROW_LEVEL_RESULT_FIELDS
            if child_excluded:
                continue
            if normalized == "success" and item is False:
                return True
            if _has_nested_success_false(item, excluded=child_excluded):
                return True
        return False
    if isinstance(value, list):
        return any(_has_nested_success_false(item, excluded=excluded) for item in value)
    if isinstance(value, str) and not excluded:
        embedded = _parse_json_value(value)
        return embedded is not None and _has_nested_success_false(embedded)
    return False


def _has_false_field(value: Any, field: str, *, excluded: bool = False) -> bool:
    target = _normalized_field_name(field)
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = _normalized_field_name(str(key))
            child_excluded = excluded or normalized in _ROW_LEVEL_RESULT_FIELDS
            if child_excluded:
                continue
            if normalized == target and item is False:
                return True
            if _has_false_field(item, field, excluded=child_excluded):
                return True
        return False
    if isinstance(value, list):
        return any(_has_false_field(item, field, excluded=excluded) for item in value)
    if isinstance(value, str) and not excluded:
        embedded = _parse_json_value(value)
        return embedded is not None and _has_false_field(embedded, field)
    return False


def _has_descendant_success_false(value: Any) -> bool:
    """扫描根对象以下任意层级的精确 ``success: false``。"""
    if isinstance(value, dict):
        return any(
            _has_nested_success_false(item)
            for key, item in value.items()
            if _normalized_field_name(str(key)) not in ({"success"} | _ROW_LEVEL_RESULT_FIELDS)
        )
    if isinstance(value, list):
        return _has_nested_success_false(value)
    return False


def _has_explicit_upload_failure_value(value: Any, *, excluded: bool = False) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = _normalized_field_name(str(key))
            child_excluded = excluded or normalized in _ROW_LEVEL_RESULT_FIELDS
            if child_excluded:
                continue
            if normalized in _FAILURE_VALUE_FIELDS and isinstance(item, str):
                if _EXPLICIT_UPLOAD_FAILURE_VALUE_PATTERN.search(item.strip()):
                    return True
            if _has_explicit_upload_failure_value(item, excluded=child_excluded):
                return True
        return False
    if isinstance(value, list):
        return any(_has_explicit_upload_failure_value(item, excluded=excluded) for item in value)
    if isinstance(value, str) and not excluded:
        embedded = _parse_json_value(value)
        return embedded is not None and _has_explicit_upload_failure_value(embedded)
    return False


def _json_like_failure_field_value(body: str) -> bool:
    if re.search(r'["\'](?:sheetData|allFileData)["\']\s*:', body, re.I):
        return False
    return any(
        _FAILURE_VALUE_PATTERN.search(match.group(1).strip())
        for match in _JSON_LIKE_FAILURE_FIELD_PATTERN.finditer(body)
        if match.group(1).strip()
    )


def _json_like_explicit_upload_failure_value(body: str) -> bool:
    if re.search(r'["\'](?:sheetData|allFileData)["\']\s*:', body, re.I):
        return False
    return any(
        _EXPLICIT_UPLOAD_FAILURE_VALUE_PATTERN.search(match.group(1).strip())
        for match in _JSON_LIKE_FAILURE_FIELD_PATTERN.finditer(body)
        if match.group(1).strip()
    )


def _nonstandard_failure_field_value(body: str) -> str:
    if re.search(r'["\'](?:sheetData|allFileData)["\']\s*[:=]', body, re.I):
        return ""
    for match in _NONSTANDARD_FAILURE_FIELD_PATTERN.finditer(body or ""):
        value = match.group(1).strip()
        if value and _FAILURE_VALUE_PATTERN.search(value):
            return value
    return ""


def _visible_html_text(body: str) -> str:
    without_scripts = re.sub(r"<(?:script|style)\b[^>]*>.*?</(?:script|style)\s*>", " ", body, flags=re.I | re.S)
    return normalize_text(html_unescape(re.sub(r"<[^>]+>", " ", without_scripts)))


def _has_non_json_failure_page(body: str) -> bool:
    text = (body or "").strip()
    if not text:
        return False
    is_html = bool(re.search(r"<(?:!doctype\s+html|html|head|body|title|h[1-6])\b", text, re.I))
    if not is_html:
        return bool(_FAILURE_VALUE_PATTERN.search(text) or _EXCEPTION_PAGE_PATTERN.search(text))
    headings = " ".join(_visible_html_text(match.group(1)) for match in _HTML_ERROR_CONTEXT_PATTERN.finditer(text))
    visible = _visible_html_text(text)
    return bool(
        (headings and (_FAILURE_VALUE_PATTERN.search(headings) or _EXCEPTION_PAGE_PATTERN.search(headings)))
        or _EXCEPTION_PAGE_PATTERN.search(visible)
        or (
            len(visible) <= 500
            and re.search(r"上传失败|保存失败|解析文件失败", visible, re.I)
            and not re.search(r"上传成功|保存成功|文件已保存", visible, re.I)
        )
    )


def _looks_like_json(body: str) -> bool:
    return _parse_json_value(body) is not None


def _parse_json_value(body: str) -> Any | None:
    value = (body or "").strip()
    if not value or value[0] not in "{[":
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _parse_json_object(body: str) -> dict[str, Any]:
    try:
        value = json.loads(body.strip())
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}
