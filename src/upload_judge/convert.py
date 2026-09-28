"""TDP 原始 JSONL 到统一 UploadEvent 结构的 UTF-8 转换器。"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import zlib
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import unquote, urlsplit

SENSITIVE_HEADER_NAMES = {
    "authorization", "cookie", "set-cookie", "x-auth-token", "x-csrf-token",
    "csrf-token", "x-xsrf-token", "token", "proxy-authorization",
}
_FILENAME_PATTERNS = [
    re.compile(r"filename\*\s*=\s*(?:UTF-8''|utf-8'')?([^;\r\n]+)", re.I),
    re.compile(r'filename\s*=\s*"([^"]+)"', re.I),
    re.compile(r"filename\s*=\s*'([^']+)'", re.I),
    re.compile(r"filename\s*=\s*([^;\r\n]+)", re.I),
    re.compile(r'["\'](?:filename|fileName|originalName)["\']\s*:\s*["\']([^"\']+)["\']', re.I),
]
_FIELD_NAME = re.compile(r"name\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^;\r\n]+))", re.I)


def parse_headers(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        source = {str(key): str(value) for key, value in raw.items()}
    elif isinstance(raw, str):
        source = {}
        for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            if ":" in line:
                name, value = line.split(":", 1)
                if name.strip():
                    source[name.strip()] = value.strip()
    else:
        source = {}
    result: dict[str, str] = {}
    for name, value in source.items():
        lowered = name.lower()
        result[name] = "<redacted>" if lowered in SENSITIVE_HEADER_NAMES or "token" in lowered or "secret" in lowered else value
    return result


def get_header(headers: dict[str, str], name: str) -> str:
    target = name.lower()
    return next((value for key, value in headers.items() if key.lower() == target), "")


def normalize_uri(raw_url: Any) -> str:
    if not isinstance(raw_url, str) or not raw_url.strip():
        return ""
    value = raw_url.strip()
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if parts.scheme and parts.netloc:
        return (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    return value


def file_ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower()[:32] if "." in filename else ""


def extract_filename_and_field(body: Any, *fallbacks: Any) -> tuple[str, str]:
    texts = [value for value in (body, *fallbacks) if isinstance(value, str)]
    filename = ""
    for text in texts:
        for pattern in _FILENAME_PATTERNS:
            match = pattern.search(text)
            if match:
                filename = _clean_filename(match.group(1))
                break
        if filename:
            break
    field_name = ""
    source = body if isinstance(body, str) else ""
    if filename and source:
        position = source.lower().find("filename")
        segment = source[max(0, position - 300): position + 300]
        match = _FIELD_NAME.search(segment)
        if match:
            field_name = next((group for group in match.groups() if group), "").strip()
    return filename, field_name


def get_http(record: dict[str, Any]) -> dict[str, Any] | None:
    value: Any = record
    for key in ("detail", "data", "detail", "alert", "net", "http"):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if isinstance(value, dict) else None


def convert_record(record: dict[str, Any], label: str, line_number: int = 0) -> dict[str, Any] | None:
    http = get_http(record)
    if not http:
        return None
    request_headers = parse_headers(http.get("reqs_header"))
    response_headers = parse_headers(http.get("resp_header"))
    request_body = http.get("reqs_body") if isinstance(http.get("reqs_body"), str) else ""
    response_body = http.get("resp_body") if isinstance(http.get("resp_body"), str) else ""
    filename, field_name = extract_filename_and_field(request_body, http.get("reqs_line"), http.get("url"))
    status_raw = str(http.get("status") or "")
    return {
        "event_id": str(record.get("id") or f"tdp_{label}_{line_number}"),
        "label": label,
        "request": {
            "method": str(http.get("method") or "POST"),
            "uri": normalize_uri(http.get("url")),
            "headers": request_headers,
            "content_type": get_header(request_headers, "content-type"),
            "filename": filename,
            "file_ext": file_ext(filename),
            "field_name": field_name,
            "body_excerpt": request_body[:500],
        },
        "response": {
            "status_code": int(status_raw) if status_raw.isdigit() else 0,
            "headers": response_headers,
            "content_type": get_header(response_headers, "content-type"),
            "body": response_body[:20000],
        },
    }


def iter_converted(path: Path, label: str) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                converted = convert_record(json.loads(line), label, line_number)
                if converted:
                    yield converted


def convert_file(input_path: Path, output_path: Path, label: str) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in iter_converted(input_path, label):
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            total += 1
    return {"input": str(input_path), "output": str(output_path), "label": label, "total": total}


def _clean_filename(value: str) -> str:
    cleaned = unquote(value.strip().strip('"\'')).replace("\\", "/").rsplit("/", 1)[-1]
    return cleaned.strip().strip(". ")[:255]


def convert_http_dump(text: str | bytes, event_id: str = "") -> dict[str, Any]:
    """将原始 HTTP 请求+响应文本转换为标准 UploadEvent dict。

    输入为告警/抓包导出的原始文本（一次请求一次响应）：
        请求行 + 请求头 + 请求体（multipart），随后是响应状态行 + 响应头 + 响应体。
    例：
        POST /api/upload HTTP/1.0
        Host: ...
        Content-Type: multipart/form-data; boundary=----x

        ------x
        Content-Disposition: form-data; name="files"; filename="a.xlsx"
        ...(body)...
        HTTP/1.1 200 OK
        Content-Type: application/json

        {"success":false,...}
    """
    if isinstance(text, bytes):
        return _convert_http_dump_bytes(text, event_id=event_id)
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    # 请求切到第一个 HTTP/ 响应，响应取最后一个：跳过 Expect:100-continue 等空中间响应
    response_matches = list(re.finditer(r"^HTTP/\d(?:\.\d)?\s+(\d{3})[^\n]*", text, re.M))
    if response_matches:
        first_match, last_match = response_matches[0], response_matches[-1]
        request_text = text[: first_match.start()]
        response_text = text[last_match.start():]
        status_code = int(last_match.group(1))
    else:
        request_text = text
        response_text = ""
        status_code = 0

    method, uri = _parse_request_line(request_text)
    request_head, request_body = _split_head_body(request_text)
    request_headers = parse_headers(request_head)
    filename, field_name = extract_filename_and_field(request_body, request_head or request_text)
    filename = _recover_filename(filename)

    response_headers: dict[str, str] = {}
    response_body = ""
    if response_text:
        response_head, response_body = _split_head_body(response_text)
        response_headers = parse_headers(response_head)
        response_body = _decompress_body(response_body, get_header(response_headers, "content-encoding"))

    derived_id = event_id or f"raw_alert_{hashlib.sha256(request_text.encode('utf-8', 'replace')).hexdigest()[:12]}"
    return {
        "event_id": derived_id,
        "request": {
            "method": method or "POST",
            "uri": normalize_uri(uri),
            "headers": request_headers,
            "content_type": get_header(request_headers, "content-type"),
            "filename": filename,
            "file_ext": file_ext(filename),
            "field_name": field_name,
            "body_excerpt": (request_body or "")[:500],
        },
        "response": {
            "status_code": status_code,
            "headers": response_headers,
            "content_type": get_header(response_headers, "content-type"),
            "body": (response_body or "")[:20000],
        },
    }


def _parse_request_line(text: str) -> tuple[str, str]:
    """在文本中定位真正的请求行（跳过文件开头的批注/说明行）。"""
    for line in text.split("\n"):
        parts = line.split()
        if len(parts) >= 3 and re.match(r"^[A-Z]+$", parts[0]) and parts[2].startswith("HTTP/"):
            return parts[0], parts[1]
    return "POST", ""


def _split_head_body(text: str) -> tuple[str, str]:
    """在请求行/状态行后切分 头部 与 体。

    兼容三种形态：
    - 标准 HTTP：头部后有空行分隔体；
    - 部分抓包导出：无空行，体直接以 multipart 边界 `--`、JSON `{`/`[`、或无冒号行开始；
    - 头部之间也被空行分隔（罕见导出格式）：跳过空行继续收集头部，到体开始才停。
    """
    lines = text.split("\n")
    boundary_index = 1
    index = 1
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            index += 1
            boundary_index = index
            continue
        if line.startswith("--") or line.startswith("{") or line.startswith("["):
            boundary_index = index
            break
        if ":" not in line:
            boundary_index = index
            break
        boundary_index = index + 1
        index += 1
    return "\n".join(lines[:boundary_index]), "\n".join(lines[boundary_index:])


def _decompress_body(body: str, content_encoding: str) -> str:
    """按 Content-Encoding 对响应体做 best-effort 解压（gzip/deflate/br）。

    由于 CLI 以文本读取原始抓包，二进制压缩体经 UTF-8 replace 后可能已损坏；
    本函数仅对可恢复的情况生效，解压失败或 brotli 未安装时原样返回。
    """
    encoding = (content_encoding or "").lower()
    if not encoding or not body:
        return body
    try:
        data = body.encode("latin-1")
        if "gzip" in encoding:
            return gzip.decompress(data).decode("utf-8", "replace")
        if "deflate" in encoding:
            try:
                return zlib.decompress(data).decode("utf-8", "replace")
            except zlib.error:
                return zlib.decompress(data, -zlib.MAX_WBITS).decode("utf-8", "replace")
        if "br" in encoding:
            try:
                import brotli  # type: ignore
            except ImportError:
                return body
            return brotli.decompress(data).decode("utf-8", "replace")
    except Exception:
        pass
    return body


def _convert_http_dump_bytes(raw: bytes, event_id: str = "") -> dict[str, Any]:
    """保留压缩响应体的原始字节，解压后再进入文本 HTTP 解析器。"""
    original = raw or b""
    response_matches = list(re.finditer(rb"^HTTP/\d(?:\.\d)?\s+(\d{3})[^\r\n]*", original, re.M))
    if not response_matches:
        return convert_http_dump(_decode_http_bytes(original), event_id=event_id)

    first_match, last_match = response_matches[0], response_matches[-1]
    request_bytes = original[: first_match.start()]
    response_bytes = original[last_match.start():]
    response_head_bytes, response_body_bytes = _split_head_body_bytes(response_bytes)
    response_head = response_head_bytes.decode("latin-1", "replace")
    response_headers = parse_headers(response_head)
    decompressed = _decompress_body_bytes(
        response_body_bytes,
        get_header(response_headers, "content-encoding"),
    )
    response_body = _decode_response_bytes(
        decompressed,
        get_header(response_headers, "content-type"),
    )
    request_text = _decode_http_bytes(request_bytes)
    separator = "" if not request_text or request_text.endswith("\n") else "\n"
    rebuilt = f"{request_text}{separator}{response_head}\n\n{response_body}"
    return convert_http_dump(rebuilt, event_id=event_id)


def _split_head_body_bytes(value: bytes) -> tuple[bytes, bytes]:
    for separator in (b"\r\n\r\n", b"\n\n", b"\r\r"):
        standard = value.find(separator)
        if standard >= 0:
            return value[:standard], value[standard + len(separator) :]

    normalized = value.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    lines = normalized.split(b"\n")
    boundary_index = 1
    for index in range(1, len(lines)):
        line = lines[index].rstrip()
        if not line:
            boundary_index = index + 1
            continue
        if line.startswith((b"--", b"{", b"[")) or b":" not in line:
            boundary_index = index
            break
        boundary_index = index + 1
    return b"\n".join(lines[:boundary_index]), b"\n".join(lines[boundary_index:])


def _decompress_body_bytes(body: bytes, content_encoding: str) -> bytes:
    data = body
    encodings = [item.strip().lower() for item in (content_encoding or "").split(",") if item.strip()]
    try:
        for encoding in reversed(encodings):
            if encoding in {"gzip", "x-gzip"}:
                data = gzip.decompress(data)
            elif encoding == "deflate":
                try:
                    data = zlib.decompress(data)
                except zlib.error:
                    data = zlib.decompress(data, -zlib.MAX_WBITS)
            elif encoding == "br":
                import brotli

                data = brotli.decompress(data)
        return data
    except Exception:
        return body


def _decode_http_bytes(value: bytes) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        return value.decode("utf-8", "replace")


def _decode_response_bytes(value: bytes, content_type: str) -> str:
    charset = re.search(r"charset\s*=\s*[\"']?([a-z0-9._-]+)", content_type or "", re.I)
    candidates = [charset.group(1)] if charset else []
    candidates.extend(["utf-8", "gb18030", "latin-1"])
    for encoding in dict.fromkeys(candidates):
        try:
            return value.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return value.decode("utf-8", "replace")


def _recover_filename(value: str) -> str:
    """对含替换符（已丢失字节）的 GBK 双解码乱码文件名做 best-effort 恢复。

    仅当文件名含 U+FFFD（乱码导致的字节丢失信号）才尝试往返；正常中文/ASCII
    文件名不做往返，避免误伤。
    """
    if not value or "�" not in value:
        return value
    try:
        candidate = value.encode("gbk", errors="ignore").decode("utf-8", errors="replace")
        if candidate.count("�") <= value.count("�"):
            return candidate
    except Exception:
        pass
    return value
