"""字符 TF-IDF + 稀疏结构化特征的二分类模型。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from training_backends.sparse_classifier import SparseClassifier
from training_core.artifacts import guard_output

from upload_judge.decision import Thresholds
from upload_judge.features import build_model_text, extract_structured_features
from upload_judge.schemas import UploadEvent
from upload_judge.versions import FEATURE_VERSION, MODEL_VERSION

SUCCESS_LABELS = {"confirmed_upload_success", "likely_upload_success", "success"}
FAILED_LABELS = {"failed", "failure"}


class UploadClassifier:
    def __init__(
        self,
        backend: str = "lightgbm",
        thresholds: Thresholds | None = None,
        random_state: int = 42,
    ) -> None:
        if backend not in {"lightgbm", "logistic"}:
            raise ValueError("backend 必须是 lightgbm 或 logistic")
        self.backend = backend
        self.thresholds = thresholds or Thresholds()
        self.random_state = random_state
        self.vectorizer: TfidfVectorizer | None = None
        self.classifier: Any = None
        self._sparse_model: SparseClassifier | None = None
        self.model_version = MODEL_VERSION
        self.manifest: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self.vectorizer is not None and self.classifier is not None

    def fit(self, events: list[UploadEvent], labels: list[str]) -> dict[str, Any]:
        if len(events) != len(labels) or not events:
            raise ValueError("events/labels 必须等长且非空")
        keep: list[int] = []
        y_values: list[int] = []
        for index, label in enumerate(labels):
            value = str(label).lower()
            if value in SUCCESS_LABELS:
                keep.append(index)
                y_values.append(1)
            elif value in FAILED_LABELS:
                keep.append(index)
                y_values.append(0)
        if len(set(y_values)) != 2:
            raise ValueError("训练数据必须同时包含 success 与 failed")
        kept_events = [events[index] for index in keep]
        texts = [build_model_text(event) for event in kept_events]
        sparse_model = SparseClassifier(backend=self.backend, random_state=self.random_state)
        sparse_model.fit(
            texts,
            np.asarray(y_values, dtype=np.int8),
            [self._struct_row(event) for event in kept_events],
        )
        self._sparse_model = sparse_model
        self.vectorizer = sparse_model.vectorizer
        self.classifier = sparse_model.classifier
        assert self.vectorizer is not None
        self.manifest = {
            "model_version": self.model_version,
            "feature_version": FEATURE_VERSION,
            "backend": self.backend,
            "random_state": self.random_state,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_samples": len(y_values),
            "training_success": int(sum(y_values)),
            "training_failed": int(len(y_values) - sum(y_values)),
            "text_features": len(self.vectorizer.vocabulary_),
            "structured_features": 24,
            "thresholds": self.thresholds.__dict__,
            "sparse_pipeline": True,
        }
        return dict(self.manifest)

    def predict_success_proba(self, events: list[UploadEvent]) -> np.ndarray:
        if not events:
            return np.asarray([], dtype=float)
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练或加载")
        model = self._shared_model()
        probabilities = model.predict_proba(
            [build_model_text(event) for event in events],
            [self._struct_row(event) for event in events],
        )
        classes = list(model.classes_)
        success_index = classes.index(1)
        return np.asarray(probabilities[:, success_index], dtype=float)

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("不能保存未训练模型")
        target = guard_output(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.backend,
            "thresholds": self.thresholds.__dict__,
            "random_state": self.random_state,
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "model_version": self.model_version,
            "manifest": self.manifest,
        }
        with target.open("xb") as handle:
            joblib.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> "UploadClassifier":
        payload = joblib.load(Path(path))
        instance = cls(
            backend=payload["backend"],
            thresholds=Thresholds(**payload.get("thresholds", {})),
            random_state=int(payload.get("random_state", 42)),
        )
        instance.vectorizer = payload["vectorizer"]
        instance.classifier = payload["classifier"]
        instance.model_version = payload.get("model_version", MODEL_VERSION)
        instance.manifest = payload.get("manifest", {})
        return instance

    def _shared_model(self) -> SparseClassifier:
        """Attach legacy fitted components without changing their artifact format."""
        if (
            self._sparse_model is None
            or self._sparse_model.vectorizer is not self.vectorizer
            or self._sparse_model.classifier is not self.classifier
        ):
            assert self.vectorizer is not None
            self._sparse_model = SparseClassifier.from_fitted(
                backend=self.backend,
                random_state=self.random_state,
                vectorizer=self.vectorizer,
                classifier=self.classifier,
                structured_features=24,
            )
        return self._sparse_model

    @staticmethod
    def _struct_matrix(events: list[UploadEvent]) -> csr_matrix:
        rows = [UploadClassifier._struct_row(event) for event in events]
        return csr_matrix(np.asarray(rows, dtype=np.float32))

    @staticmethod
    def _struct_row(event: UploadEvent) -> list[float]:
        f = extract_structured_features(event)
        sf, ff = set(f["success_fields"]), set(f["fail_fields"])
        ext = f["file_ext"]
        return [
            min(max(f["status_code"], 0), 999) / 999.0,
            float(200 <= f["status_code"] < 300),
            min(np.log1p(len(f["body"])) / 12.0, 1.0),
            float(f["is_json"]),
            float(f["is_html"]),
            float(f["body_empty"]),
            float(bool(f["paths"])),
            min(len(f["paths"]) / 5.0, 1.0),
            float(bool(f["file_ids"])),
            min(len(f["file_ids"]) / 5.0, 1.0),
            float(f["filename_in_response"]),
            min(len(event.request.filename) / 100.0, 1.0),
            min(len(ext) / 10.0, 1.0),
            float(bool(f["location"])),
            min(len(sf) / 4.0, 1.0),
            min(len(ff) / 3.0, 1.0),
            float("success=true" in sf),
            float("code=0" in sf),
            float("state=success" in sf),
            float("ok=true" in sf),
            float("upload_success_text" in sf),
            float("success=false" in ff),
            float(bool(ff)),
            min(len(event.request.uri) / 500.0, 1.0),
        ]
