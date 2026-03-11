"""Shared datatypes and custom estimators for rating experiments."""

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression


@dataclass
class ModelSpec:
    """Declarative model specification used by the training runner."""

    name: str
    family: str  # classifier | classifier_dense | regressor | regressor_dense | ordinal
    build: Callable[[], object]


@dataclass
class FoldEval:
    """Per-fold evaluation metrics for one model and one feature dimension."""

    fold: int
    k: int
    model: str
    accuracy: float
    macro_f1: float
    mae: float


@dataclass
class RunConfig:
    """Runtime configuration for dataset paths, target column, and outputs."""

    input_csv: str
    output_csv: str
    output_cv_folds_csv: str
    output_model_summary_csv: str
    target_col: str
    k_values: List[int]
    lsvc_safe_k_values: List[int]
    test_size: float
    cv_splits: int
    lsvc_safe_cv_splits: int
    random_state: int
    max_rows: int | None
    requested_models: List[str] | None
    include_test_row: bool


class OrdinalLogisticClassifier:
    """Cumulative-link style ordinal classifier using one-vs-threshold logits."""

    def __init__(self, min_label: int = 1, max_label: int = 5, random_state: int = 42):
        """Initialize ordinal label bounds and internal model containers."""
        self.min_label = min_label
        self.max_label = max_label
        self.random_state = random_state
        self.thresholds_: List[int] = []
        self.models_: List[object] = []
        self.constants_: List[float] = []

    def fit(self, x: csr_matrix, y: np.ndarray) -> "OrdinalLogisticClassifier":
        """Fit one binary logistic model per cumulative threshold."""
        self.thresholds_ = list(range(self.min_label, self.max_label))
        self.models_ = []
        self.constants_ = []

        for t in self.thresholds_:
            y_bin = (y > t).astype(int)
            clf = LogisticRegression(
                solver="saga",
                penalty="l2",
                max_iter=2000,
                random_state=self.random_state,
                n_jobs=-1,
            )
            clf.fit(x, y_bin)
            self.models_.append(clf)
            self.constants_.append(-1.0)

        return self

    def _p_gt_thresholds(self, x: csr_matrix) -> np.ndarray:
        """Predict probability of class labels being above each threshold."""
        probs = []
        for model, const in zip(self.models_, self.constants_):
            p = model.predict_proba(x)[:, 1]
            probs.append(np.clip(p, 0.0, 1.0))
        return np.column_stack(probs)

    def predict(self, x: csr_matrix) -> np.ndarray:
        """Predict the most likely ordinal class for each input row."""
        p_gt = self._p_gt_thresholds(x)
        n = x.shape[0]
        n_classes = self.max_label - self.min_label + 1
        probs = np.zeros((n, n_classes), dtype=float)

        probs[:, 0] = 1.0 - p_gt[:, 0]
        for i in range(1, n_classes - 1):
            probs[:, i] = p_gt[:, i - 1] - p_gt[:, i]
        probs[:, -1] = p_gt[:, -1]

        probs = np.clip(probs, 0.0, 1.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        probs = probs / row_sums

        preds = probs.argmax(axis=1) + self.min_label
        return preds.astype(int)
