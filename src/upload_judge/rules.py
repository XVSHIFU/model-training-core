"""只处理无歧义的强失败和强成功规则。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from upload_judge.evidence import extract_evidence, has_failure_semantics, has_success_semantics
from upload_judge.features import extract_structured_features
from upload_judge.schemas import Evidence, UploadEvent, Verdict
from upload_judge.versions import RULE_VERSION

SCRIPT_LIKE_EXTENSIONS = {
    "php", "php3", "php4", "php5", "php7", "phtml", "phar",
    "jsp", "jspx", "jspf", "asp", "aspx", "ashx", "asa", "cer",
    "war", "jar", "exe", "dll", "sh", "bat", "cmd", "ps1", "cgi", "pl",
}
SAFE_EXTENSIONS = {
    "step", "stp", "stl", "obj", "dxf", "dwg", "igs", "iges", "glb",
    "gltf", "fbx", "skp", "sldprt", "sldasm",
}


@dataclass(frozen=True)
class RuleDecision:
    verdict: Verdict
    confidence: float
    reason: str
    rule_name: str
    evidence: list[Evidence] = field(default_factory=list)
    negative_evidence: list[Evidence] = field(default_factory=list)


def apply_rules(event: UploadEvent) -> RuleDecision | None:
    positive, negative = extract_evidence(event)
    features = extract_structured_features(event)
    body = features["body_lower"]
    login_page = features["is_html"] and bool(
        re.search(
            r"<(?:title|h[1-6])\b[^>]*>[^<]*(?:login|sign\s*in|登录|用户登录)[^<]*</"
            r"|<input\b[^>]*\btype\s*=\s*[\"']?password\b",
            body,
            re.I,
        )
    )
    permission_status = features["status_code"] in {401, 403} and any(
        marker in body for marker in ("unauthorized", "forbidden", "permission", "access denied", "未授权", "权限", "拒绝")
    )
    failure_for_rule = has_failure_semantics(negative) and not features["weak_failure_conflict"]
    if failure_for_rule or login_page or permission_status:
        if login_page:
            negative.append(Evidence(type="failure_field", value="login_page"))
        if permission_status:
            negative.append(Evidence(type="status_code", value=str(features["status_code"])))
        return RuleDecision(
            verdict=Verdict.FAILED,
            confidence=0.97,
            evidence=positive,
            negative_evidence=negative,
            reason="响应包含明确失败、拒绝、拦截、鉴权失败或登录页证据",
            rule_name="strong_failure",
        )

    ext = features["file_ext"]
    weak_body = body.strip() in {"ok", "success", "true"}
    if (
        ext not in SCRIPT_LIKE_EXTENSIONS
        and not features["body_empty"]
        and not weak_body
        and has_success_semantics(positive)
        and features["success_strength"] == "explicit"
        and features["evidence_grade"] == "STRONG"
    ):
        return RuleDecision(
            verdict=Verdict.CONFIRMED_UPLOAD_SUCCESS,
            confidence=0.97,
            evidence=positive,
            negative_evidence=negative,
            reason="响应包含明确上传成功语义及路径、文件 ID、Location 或文件名回显",
            rule_name="strong_success",
        )
    return None
