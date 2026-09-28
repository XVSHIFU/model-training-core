"""规则与模型的唯一研判编排；single 直接复用 batch。"""

from __future__ import annotations

from pathlib import Path

from upload_judge.decision import decide
from upload_judge.evidence import extract_evidence
from upload_judge.ml_model import UploadClassifier
from upload_judge.rules import apply_rules
from upload_judge.schemas import JudgmentResult, UploadEvent, Verdict
from upload_judge.versions import FEATURE_VERSION, RULE_VERSION


class UploadJudge:
    def __init__(self, model_path: str | Path | None = None, classifier: UploadClassifier | None = None) -> None:
        if classifier is None and model_path is None:
            raise ValueError("model_path 与 classifier 至少提供一个")
        self.classifier = classifier or UploadClassifier.load(model_path)  # type: ignore[arg-type]

    def judge(self, event: UploadEvent) -> JudgmentResult:
        return self.judge_batch([event])[0]

    def judge_batch(self, events: list[UploadEvent]) -> list[JudgmentResult]:
        results: list[JudgmentResult | None] = [None] * len(events)
        model_indices: list[int] = []
        model_events: list[UploadEvent] = []
        cached_evidence: dict[int, tuple] = {}

        for index, event in enumerate(events):
            positive, negative = extract_evidence(event)
            cached_evidence[index] = (positive, negative)
            rule = apply_rules(event)
            if rule is None:
                model_indices.append(index)
                model_events.append(event)
                continue
            results[index] = self._result(
                event,
                rule.verdict,
                rule.confidence,
                rule.evidence or positive,
                rule.negative_evidence or negative,
                rule.reason,
                {"source": "rule", "rule": rule.rule_name},
            )

        probabilities = self.classifier.predict_success_proba(model_events)
        for index, event, probability in zip(model_indices, model_events, probabilities):
            positive, negative = cached_evidence[index]
            resolved = decide(event, float(probability), positive, negative, self.classifier.thresholds)
            results[index] = self._result(
                event,
                resolved.verdict,
                resolved.confidence,
                positive,
                negative,
                resolved.reason,
                {"source": resolved.source, "p_success": round(float(probability), 6)},
            )

        return [result for result in results if result is not None]

    def _result(self, event, verdict, confidence, positive, negative, reason, meta) -> JudgmentResult:
        return JudgmentResult(
            event_id=event.event_id,
            verdict=verdict,
            upload_saved=verdict in {Verdict.CONFIRMED_UPLOAD_SUCCESS, Verdict.LIKELY_UPLOAD_SUCCESS},
            confidence=round(min(1.0, max(0.0, float(confidence))), 4),
            evidence=positive,
            negative_evidence=negative,
            reason=reason,
            rule_version=RULE_VERSION,
            model_version=self.classifier.model_version,
            feature_version=FEATURE_VERSION,
            meta=meta,
        )
