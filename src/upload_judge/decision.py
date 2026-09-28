"""所有模型概率到四类 verdict 的唯一纯决策函数。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from upload_judge.evidence import has_failure_semantics, has_success_semantics
from upload_judge.features import extract_structured_features
from upload_judge.rules import SAFE_EXTENSIONS
from upload_judge.schemas import Evidence, UploadEvent, Verdict


@dataclass(frozen=True)
class Thresholds:
    confirmed: float = 0.85
    likely: float = 0.60
    failed: float = 0.85

    def validate(self) -> None:
        if not 0 <= self.likely <= self.confirmed <= 1 or not 0 <= self.failed <= 1:
            raise ValueError("阈值必须满足 0 <= likely <= confirmed <= 1 且 0 <= failed <= 1")


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    confidence: float
    reason: str
    source: str


def _decision(
    verdict: Verdict,
    confidence: float,
    reason: str,
    source: str,
) -> Decision:
    return Decision(verdict, confidence, reason, source)


def decide(
    event: UploadEvent,
    p_success: float,
    positive: list[Evidence],
    negative: list[Evidence],
    thresholds: Thresholds = Thresholds(),
) -> Decision:
    thresholds.validate()
    p_success = min(1.0, max(0.0, float(p_success)))
    p_failed = 1.0 - p_success
    features = extract_structured_features(event)
    has_success = has_success_semantics(positive)
    has_failure = has_failure_semantics(negative)
    evidence_grade = features["evidence_grade"]
    nested_business_false = "nested_success=false" in features["fail_fields"]

    if has_failure and features["strong_failure"]:
        return _decision(Verdict.FAILED, max(p_failed, 0.97), "存在确定性强失败证据", "evidence_strong_failure")
    if has_failure and has_success:
        return _decision(Verdict.UNKNOWN, 0.5, "上传相关的强成功与强失败证据冲突", "evidence_conflict")
    if has_failure:
        return _decision(Verdict.FAILED, max(p_failed, 0.9), "存在明确失败语义", "evidence_failure")
    if nested_business_false and has_success:
        return _decision(Verdict.UNKNOWN, 0.5, "顶层成功字段与嵌套业务失败状态冲突", "nested_evidence_conflict")

    ext = (event.request.file_ext or "").lower().strip(".")
    if ext in SAFE_EXTENSIONS and p_success >= thresholds.likely:
        return _decision(
            Verdict.UNKNOWN,
            min(p_success, 0.5),
            "业务型扩展名仅有模型倾向、缺少可独立确认的保存证据",
            "safe_ext_downgrade",
        )

    if p_success >= thresholds.confirmed:
        if (
            "upload_success_text" in features["success_fields"]
            and features["filename_in_response"]
        ):
            return _decision(
                Verdict.CONFIRMED_UPLOAD_SUCCESS,
                p_success,
                "高成功概率、明确上传成功文本且响应回显文件名",
                "model_confirmed_success_text_filename_echo",
            )
        if evidence_grade == "STRONG":
            return _decision(Verdict.CONFIRMED_UPLOAD_SUCCESS, p_success, "高成功概率且证据等级为 STRONG", "model_confirmed_strong")
        if evidence_grade == "WEAK":
            return _decision(Verdict.LIKELY_UPLOAD_SUCCESS, p_success, "高成功概率且证据等级为 WEAK", "model_likely_weak")
        if has_success and features["filename_context"]:
            return _decision(
                Verdict.LIKELY_UPLOAD_SUCCESS,
                p_success,
                "高成功概率、存在成功语义和请求文件名上下文，但无独立资源证据",
                "model_likely_success_context",
            )
        if (
            features["is_html"]
            and features["filename_context"]
            and re.search(r"<h[1-6]\b[^>]*>\s*已处理表单\s*</h[1-6]\s*>", features["body"], re.I)
        ):
            return _decision(
                Verdict.LIKELY_UPLOAD_SUCCESS,
                p_success,
                "高成功概率且上传响应明确显示表单已处理，但无独立资源回显",
                "model_likely_processed_form_ack",
            )
        return _decision(Verdict.UNKNOWN, min(p_success, 0.5), "模型倾向成功但证据等级为 NONE", "model_no_evidence")

    if p_success >= thresholds.likely and evidence_grade in {"STRONG", "WEAK"}:
        return _decision(Verdict.LIKELY_UPLOAD_SUCCESS, p_success, "中等成功概率且至少有 WEAK 证据", "model_likely_evidence")

    if p_failed >= thresholds.failed:
        return _decision(Verdict.UNKNOWN, p_failed, "模型倾向失败但缺少明确失败证据", "model_failed_without_evidence")

    return _decision(Verdict.UNKNOWN, max(p_success, p_failed), "模型概率与证据均不足", "model_unknown")
