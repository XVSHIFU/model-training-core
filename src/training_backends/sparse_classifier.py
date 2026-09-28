"""Character TF-IDF with optional external numeric features for classification.

This backend knows only text, numeric rows and discrete class labels. Task
adapters own parsing, target mapping, feature meaning and decision policies.
"""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from training_core.artifacts import guard_output

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - reported when that backend is selected
    LGBMClassifier = None  # type: ignore[assignment]


class SparseClassifier:
    FORMAT = "training_backends.sparse_classifier.v1"

    def __init__(
        self,
        backend: str = "lightgbm",
        random_state: int = 42,
        vectorizer_params: dict[str, Any] | None = None,
        classifier_params: dict[str, Any] | None = None,
    ) -> None:
        if backend not in {"lightgbm", "logistic"}:
            raise ValueError("backend must be lightgbm or logistic")
        self.backend = backend
        self.random_state = random_state
        self.vectorizer_params = dict(vectorizer_params or {})
        self.classifier_params = dict(classifier_params or {})
        self.vectorizer: TfidfVectorizer | None = None
        self.classifier: Any = None
        self.text_features_: int = 0
        self.structured_features_: int = 0
        self.n_features_in_: int = 0
        self.manifest: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self.vectorizer is not None and self.classifier is not None

    @property
    def classes_(self) -> np.ndarray:
        self._require_fitted()
        return np.asarray(self.classifier.classes_)

    def fit(
        self,
        texts: list[str],
        targets: list,
        structured: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        self._validate_texts(texts)
        if not len(texts) or len(texts) != len(targets):
            raise ValueError("texts/targets must have equal, nonzero lengths")
        y = self._validate_targets(targets)
        struct_matrix = self._structured_matrix(structured, len(texts))
        vectorizer = TfidfVectorizer(**{
            "analyzer": "char_wb",
            "ngram_range": (2, 4),
            "min_df": 2,
            "max_df": 0.995,
            "max_features": 20000,
            "sublinear_tf": True,
            "dtype": np.float32,
            **self.vectorizer_params,
        })
        text_matrix = vectorizer.fit_transform(texts)
        matrix = hstack([text_matrix, struct_matrix], format="csr", dtype=np.float32)
        classifier = self._new_classifier()
        classifier.fit(matrix, y)
        # Publish the fitted state together only after a successful fit.
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.text_features_ = text_matrix.shape[1]
        self.structured_features_ = struct_matrix.shape[1]
        self.n_features_in_ = matrix.shape[1]
        self.manifest = {
            "backend": self.backend,
            "random_state": self.random_state,
            "training_samples": len(texts),
            "classes": self.classes_.tolist(),
            "class_counts": [int(np.count_nonzero(y == label)) for label in self.classes_],
            "text_features": self.text_features_,
            "structured_features": self.structured_features_,
            "total_features": self.n_features_in_,
            "sparse_pipeline": True,
        }
        return dict(self.manifest)

    def predict_proba(
        self, texts: list[str], structured: list[list[float]] | None = None
    ) -> np.ndarray:
        matrix = self._prediction_matrix(texts, structured)
        if not len(texts):
            return np.empty((0, len(self.classes_)), dtype=float)
        return np.asarray(self.classifier.predict_proba(matrix), dtype=float)

    def predict(
        self, texts: list[str], structured: list[list[float]] | None = None
    ) -> np.ndarray:
        matrix = self._prediction_matrix(texts, structured)
        if not len(texts):
            return np.asarray([], dtype=self.classes_.dtype)
        return np.asarray(self.classifier.predict(matrix))

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        target = guard_output(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": self.FORMAT,
            "backend": self.backend,
            "random_state": self.random_state,
            "vectorizer_params": self.vectorizer_params,
            "classifier_params": self.classifier_params,
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "text_features": self.text_features_,
            "structured_features": self.structured_features_,
            "total_features": self.n_features_in_,
            "manifest": self.manifest,
        }
        with target.open("xb") as handle:
            joblib.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> "SparseClassifier":
        # Like sklearn/joblib, load only artifacts from a trusted source.
        payload = joblib.load(Path(path))
        if not isinstance(payload, dict) or payload.get("format") != cls.FORMAT:
            raise ValueError("Unsupported sparse classifier artifact format")
        model = cls.from_fitted(
            backend=payload["backend"],
            vectorizer=payload["vectorizer"],
            classifier=payload["classifier"],
            structured_features=payload["structured_features"],
            random_state=payload["random_state"],
            vectorizer_params=payload.get("vectorizer_params"),
            classifier_params=payload.get("classifier_params"),
        )
        if (payload["text_features"] != model.text_features_
                or payload["total_features"] != model.n_features_in_):
            raise ValueError("Artifact feature dimensions are inconsistent")
        model.manifest = payload.get("manifest", {})
        return model

    @classmethod
    def from_fitted(
        cls, *, backend: str, vectorizer: TfidfVectorizer, classifier: Any,
        structured_features: int, random_state: int = 42,
        vectorizer_params: dict[str, Any] | None = None,
        classifier_params: dict[str, Any] | None = None,
    ) -> "SparseClassifier":
        """Use existing fitted components without fitting or altering them."""
        if not isinstance(structured_features, Integral) or structured_features < 0:
            raise ValueError("structured_features must be a nonnegative integer")
        if not hasattr(vectorizer, "vocabulary_") or not hasattr(classifier, "classes_"):
            raise ValueError("Both components must already be fitted")
        model = cls(backend, random_state, vectorizer_params, classifier_params)
        model.vectorizer = vectorizer
        model.classifier = classifier
        model.text_features_ = len(vectorizer.vocabulary_)
        model.structured_features_ = int(structured_features)
        model.n_features_in_ = model.text_features_ + model.structured_features_
        if int(classifier.n_features_in_) != model.n_features_in_:
            raise ValueError("Fitted components have incompatible feature dimensions")
        return model

    def _new_classifier(self) -> Any:
        if self.backend == "logistic":
            return LogisticRegression(**{
                "C": 2.0,
                "max_iter": 1000,
                "random_state": self.random_state,
                "n_jobs": -1,
                **self.classifier_params,
            })
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed")
        return LGBMClassifier(**{
            "n_estimators": 300,
            "max_depth": 8,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": -1,
            **self.classifier_params,
        })

    def _prediction_matrix(self, texts: list[str], structured) -> csr_matrix:
        self._require_fitted()
        self._validate_texts(texts)
        struct_matrix = self._structured_matrix(structured, len(texts), self.structured_features_)
        if not len(texts):
            return csr_matrix((0, self.n_features_in_), dtype=np.float32)
        assert self.vectorizer is not None
        matrix = hstack(
            [self.vectorizer.transform(texts), struct_matrix], format="csr", dtype=np.float32
        )
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature dimensions differ from training")
        return matrix

    @staticmethod
    def _structured_matrix(structured, rows: int, expected_width: int | None = None) -> csr_matrix:
        if structured is None:
            if rows and expected_width not in {None, 0}:
                raise ValueError("Structured features used during training are required")
            return csr_matrix((rows, expected_width or 0), dtype=np.float32)
        try:
            array = np.asarray(structured, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Structured features must be a rectangular numeric matrix") from exc
        if rows == 0 and array.size == 0 and array.ndim == 1:
            array = array.reshape(0, expected_width or 0)
        if array.ndim != 2 or array.shape[0] != rows:
            raise ValueError("Structured features must have one equally sized row per text")
        if expected_width is not None and array.shape[1] != expected_width:
            raise ValueError("Structured feature width differs from training")
        if not np.isfinite(array).all():
            raise ValueError("Structured features must contain finite values")
        return csr_matrix(array, dtype=np.float32)

    @staticmethod
    def _validate_texts(texts) -> None:
        if isinstance(texts, (str, bytes)) or not all(isinstance(text, str) for text in texts):
            raise ValueError("texts must be a sequence of strings")

    @staticmethod
    def _validate_targets(targets) -> np.ndarray:
        string_labels = all(isinstance(label, str) and bool(label.strip()) for label in targets)
        integer_labels = all(isinstance(label, Integral) for label in targets)
        if not (string_labels or integer_labels):
            raise ValueError("Targets must be nonempty strings or integer class labels of one type")
        values = np.asarray(targets)
        if values.ndim != 1 or len(np.unique(values)) < 2:
            raise ValueError("Training requires at least two distinct classes")
        return values

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Classifier is not fitted")
