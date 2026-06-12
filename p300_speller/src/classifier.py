"""classifier.py — the P300 target/non-target discriminator.

A P300 speller reduces to a binary detection problem: for each row/column
flash, did the elicited epoch contain a P300 (the user attended a symbol in that
row/column) or not? The intersection of the highest-scoring row and column then
identifies the intended character.

This module wraps a scikit-learn :class:`~sklearn.pipeline.Pipeline` of
``StandardScaler`` followed by a linear discriminant model. Two backends are
offered:

* **LDA** (default) — :class:`~sklearn.discriminant_analysis.LinearDiscriminant\
Analysis` with Ledoit-Wolf shrinkage, the workhorse of ERP-based BCIs because it
  is closed-form, fast, and well-behaved with few trials.
* **SVM** — :class:`~sklearn.svm.LinearSVC`, a max-margin alternative with an
  explicit regularisation knob.

Both expose a real-valued *score* via ``decision_function`` (signed distance to
the decision boundary). The pipeline averages these scores across the repeated
flashes of a character, which is the averaging step that drives the system from
single-trial chance toward usable accuracy.

The trained pipeline (including the fitted scaler and any decimation-implied
feature layout) is persisted with :mod:`joblib`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Integer label convention used throughout the pipeline.
NON_TARGET = 0
TARGET = 1


@dataclass
class TrainingReport:
    """Summary statistics from a training run.

    Attributes:
        n_samples: Total epochs used.
        n_targets: Number of target epochs.
        train_accuracy: Resubstitution accuracy (optimistic; sanity check only).
        cv_auc_mean: Mean cross-validated ROC-AUC (the honest metric).
        cv_auc_std: Standard deviation of cross-validated ROC-AUC.
    """

    n_samples: int
    n_targets: int
    train_accuracy: float
    cv_auc_mean: float
    cv_auc_std: float

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"epochs={self.n_samples} (targets={self.n_targets}) | "
            f"train_acc={self.train_accuracy:.3f} | "
            f"CV AUC={self.cv_auc_mean:.3f}+/-{self.cv_auc_std:.3f}"
        )


class P300Classifier:
    """Binary P300 detector with score-level character decoding.

    Args:
        model_type: ``"lda"`` or ``"svm"``.
        lda_shrinkage: Shrinkage for LDA: ``"auto"``, ``None``, or a float in
            ``[0, 1]``. Ignored for SVM.
        svm_c: Inverse regularisation strength for LinearSVC. Ignored for LDA.
        class_weight: Passed through to the estimator to offset the ~5:1
            non-target/target imbalance (``"balanced"`` recommended).
    """

    def __init__(
        self,
        model_type: str = "lda",
        lda_shrinkage: Optional[object] = "auto",
        svm_c: float = 0.1,
        class_weight: Optional[object] = "balanced",
    ) -> None:
        self.model_type = model_type.lower()
        self.lda_shrinkage = lda_shrinkage
        self.svm_c = svm_c
        self.class_weight = class_weight
        self.pipeline: Pipeline = self._build_pipeline()
        self._fitted = False

    # -- construction ------------------------------------------------------- #
    def _build_pipeline(self) -> Pipeline:
        """Assemble the scaler + estimator pipeline."""
        if self.model_type == "lda":
            # 'lsqr' solver is required to use shrinkage; it is exact for LDA.
            estimator = LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage=self.lda_shrinkage
            )
        elif self.model_type == "svm":
            estimator = LinearSVC(
                C=self.svm_c,
                class_weight=self.class_weight,
                dual="auto",
                max_iter=10000,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type!r}")

        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", estimator),
            ]
        )

    # -- training ----------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> TrainingReport:
        """Fit the pipeline and report cross-validated performance.

        Args:
            X: Design matrix, shape ``(n_epochs, n_features)``.
            y: Binary labels (``TARGET`` / ``NON_TARGET``), shape ``(n_epochs,)``.
            cv_folds: Number of stratified CV folds for the AUC estimate.

        Returns:
            A :class:`TrainingReport`.

        Raises:
            ValueError: If both classes are not present in ``y``.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2-D design matrix")
        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError(
                "Training data contains a single class; both target and "
                "non-target epochs are required."
            )

        # Honest performance estimate before refitting on all data.
        cv_auc_mean, cv_auc_std = self._cross_val_auc(X, y, cv_folds)

        self.pipeline.fit(X, y)
        self._fitted = True

        train_pred = self.pipeline.predict(X)
        report = TrainingReport(
            n_samples=int(X.shape[0]),
            n_targets=int(np.sum(y == TARGET)),
            train_accuracy=float(accuracy_score(y, train_pred)),
            cv_auc_mean=cv_auc_mean,
            cv_auc_std=cv_auc_std,
        )
        return report

    def _cross_val_auc(
        self, X: np.ndarray, y: np.ndarray, cv_folds: int
    ) -> Tuple[float, float]:
        """Compute mean/std cross-validated ROC-AUC, robustly to tiny sets."""
        # Need at least cv_folds members in the minority class for stratified CV.
        min_class = int(np.min(np.bincount(y)))
        folds = max(2, min(cv_folds, min_class))
        if min_class < 2:
            return float("nan"), float("nan")
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        try:
            scores = cross_val_score(
                self._build_pipeline(), X, y, cv=skf, scoring="roc_auc"
            )
        except Exception:
            return float("nan"), float("nan")
        return float(np.mean(scores)), float(np.std(scores))

    # -- inference ---------------------------------------------------------- #
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Classifier is not trained; call fit() or load().")

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """Return signed decision scores (higher = more P300-like).

        Args:
            X: Design matrix, shape ``(n_epochs, n_features)``.

        Returns:
            Score vector, shape ``(n_epochs,)``.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        # Both LDA and LinearSVC expose decision_function for binary problems.
        return np.ravel(self.pipeline.decision_function(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard binary prediction (``TARGET`` / ``NON_TARGET``)."""
        self._check_fitted()
        return self.pipeline.predict(np.asarray(X, dtype=np.float64))

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str) -> None:
        """Persist the fitted pipeline and configuration to ``path``."""
        self._check_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "model_type": self.model_type,
                "lda_shrinkage": self.lda_shrinkage,
                "svm_c": self.svm_c,
                "class_weight": self.class_weight,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "P300Classifier":
        """Load a classifier previously saved with :meth:`save`.

        Args:
            path: Path to the ``.joblib`` artifact.

        Returns:
            A ready-to-use :class:`P300Classifier`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No trained model at {path!r}")
        blob = joblib.load(path)
        obj = cls(
            model_type=blob["model_type"],
            lda_shrinkage=blob.get("lda_shrinkage", "auto"),
            svm_c=blob.get("svm_c", 0.1),
            class_weight=blob.get("class_weight", "balanced"),
        )
        obj.pipeline = blob["pipeline"]
        obj._fitted = True
        return obj


# --------------------------------------------------------------------------- #
# Character decoding from per-flash scores
# --------------------------------------------------------------------------- #
def decode_character_scores(
    row_scores: np.ndarray, col_scores: np.ndarray
) -> Tuple[int, int]:
    """Pick the target row and column from accumulated flash scores.

    Args:
        row_scores: Summed/averaged decision scores per row, shape ``(n_rows,)``.
        col_scores: Summed/averaged decision scores per col, shape ``(n_cols,)``.

    Returns:
        ``(row_index, col_index)`` of the most P300-like row and column.
    """
    return int(np.argmax(row_scores)), int(np.argmax(col_scores))
