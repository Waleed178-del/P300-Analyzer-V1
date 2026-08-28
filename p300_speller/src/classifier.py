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

**Validation policy.** Performance is estimated with *grouped* cross-validation
(:class:`~sklearn.model_selection.GroupKFold`), where a group is one acquisition
run — in practice one cued character block. The twelve flashes inside a block
share electrode drift, impedance, and the user's attentional state, so they are
not statistically independent. Splitting them across train and test folds lets
the classifier recognise the block's fingerprint rather than the shape of the
neural response, which inflates the score without improving decoding. A shuffled
:class:`~sklearn.model_selection.StratifiedKFold` estimate is still computed, but
only as a *control*: the difference between the two is reported as the optimism
introduced by ignoring group structure, and the shuffled figure is never a
result. A single pooled ROC is built from out-of-fold decision values via
:func:`~sklearn.model_selection.cross_val_predict` rather than averaging
per-fold AUCs, so folds with few targets cannot distort the estimate.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Integer label convention used throughout the pipeline.
NON_TARGET = 0
TARGET = 1

# Grouped AUC above this value on a sparse (1-3 channel) montage is not skill.
# Realistic three-channel single-trial performance sits near 0.65-0.85; a score
# above the gate indicates label or group leakage and must be investigated as a
# defect before the number is quoted anywhere.
DEFAULT_LEAKAGE_GATE_AUC = 0.95


@dataclass
class TrainingReport:
    """Summary statistics from a training run.

    Attributes:
        n_samples: Total epochs used.
        n_targets: Number of target epochs.
        train_accuracy: Resubstitution accuracy (optimistic; sanity check only).
        grouped_auc: Pooled out-of-fold ROC-AUC under :class:`GroupKFold`. This
            is the only figure that may be reported as a result.
        shuffled_auc: Pooled out-of-fold ROC-AUC under a shuffled
            :class:`StratifiedKFold`. Diagnostic control only — never a result.
        optimism: ``shuffled_auc - grouped_auc``; the inflation attributable to
            breaking group structure.
        n_groups: Number of distinct acquisition groups (character blocks).
        grouped_folds: Number of folds actually used for the grouped estimate.
        leakage_flag: ``True`` when ``grouped_auc`` exceeds the leakage gate,
            i.e. the result is implausibly good and should be treated as a bug.
    """

    n_samples: int
    n_targets: int
    train_accuracy: float
    grouped_auc: float
    shuffled_auc: float
    optimism: float
    n_groups: int
    grouped_folds: int
    leakage_flag: bool

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        head = (
            f"epochs={self.n_samples} (targets={self.n_targets}) | "
            f"train_acc={self.train_accuracy:.3f} | "
            f"grouped AUC={self.grouped_auc:.3f} "
            f"({self.grouped_folds} folds over {self.n_groups} groups)"
        )
        if not np.isnan(self.shuffled_auc):
            head += (
                f" | shuffled AUC={self.shuffled_auc:.3f} (control) "
                f"| optimism={self.optimism:+.3f}"
            )
        if self.leakage_flag:
            head += " | *** LEAKAGE GATE TRIPPED ***"
        return head


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
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[Sequence[int]] = None,
        cv_folds: int = 5,
        leakage_gate_auc: float = DEFAULT_LEAKAGE_GATE_AUC,
    ) -> TrainingReport:
        """Fit the pipeline and report grouped cross-validated performance.

        Args:
            X: Design matrix, shape ``(n_epochs, n_features)``.
            y: Binary labels (``TARGET`` / ``NON_TARGET``), shape ``(n_epochs,)``.
            groups: Acquisition-run identifier per epoch, shape ``(n_epochs,)``.
                One group per cued character block. When ``None`` — or when
                fewer than two distinct groups are present — the grouped
                estimate is undefined, ``grouped_auc`` is ``NaN``, and a warning
                is issued. Never substitute the shuffled figure in that case.
            cv_folds: Requested number of CV folds. The grouped estimate uses
                ``min(cv_folds, n_groups)``.
            leakage_gate_auc: Grouped AUC above which the run is flagged as a
                probable leakage defect rather than a result.

        Returns:
            A :class:`TrainingReport`.

        Raises:
            ValueError: If both classes are not present in ``y``, or if
                ``groups`` is given with a length that does not match ``X``.
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
        if groups is not None:
            groups = np.asarray(groups)
            if groups.shape[0] != X.shape[0]:
                raise ValueError(
                    f"groups has length {groups.shape[0]} but X has "
                    f"{X.shape[0]} rows; one group id per epoch is required."
                )

        # Honest (grouped) estimate plus its shuffled control, computed before
        # refitting on all data.
        grouped_auc, grouped_folds, n_groups = self._grouped_auc(
            X, y, groups, cv_folds
        )
        shuffled_auc = self._shuffled_auc(X, y, cv_folds)
        optimism = float(shuffled_auc - grouped_auc)

        leakage_flag = bool(
            not np.isnan(grouped_auc) and grouped_auc > leakage_gate_auc
        )
        if leakage_flag:
            warnings.warn(
                f"Grouped AUC {grouped_auc:.3f} exceeds the leakage gate "
                f"({leakage_gate_auc:.2f}). On a sparse montage this is not "
                "skill: investigate group assignment, label construction, and "
                "epoch overlap before quoting this number.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.pipeline.fit(X, y)
        self._fitted = True

        train_pred = self.pipeline.predict(X)
        report = TrainingReport(
            n_samples=int(X.shape[0]),
            n_targets=int(np.sum(y == TARGET)),
            train_accuracy=float(accuracy_score(y, train_pred)),
            grouped_auc=grouped_auc,
            shuffled_auc=shuffled_auc,
            optimism=optimism,
            n_groups=n_groups,
            grouped_folds=grouped_folds,
            leakage_flag=leakage_flag,
        )
        return report

    def _pooled_auc(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv,
        groups: Optional[np.ndarray] = None,
    ) -> float:
        """Pooled out-of-fold ROC-AUC over a single concatenated score vector.

        Every epoch is scored exactly once, while held out. Building one ROC
        from the pooled scores avoids averaging per-fold AUCs, which is unstable
        when a fold contains only a handful of targets.
        """
        try:
            scores = cross_val_predict(
                self._build_pipeline(),
                X,
                y,
                groups=groups,
                cv=cv,
                method="decision_function",
            )
            return float(roc_auc_score(y, np.ravel(scores)))
        except Exception:
            return float("nan")

    def _grouped_auc(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray],
        cv_folds: int,
    ) -> Tuple[float, int, int]:
        """Grouped (leakage-free) pooled AUC.

        Returns:
            ``(auc, n_folds_used, n_groups)``. ``auc`` is ``NaN`` and a warning
            is issued when fewer than two groups are available, because a
            grouped estimate is then undefined.
        """
        if groups is None:
            warnings.warn(
                "No acquisition groups supplied; the grouped (leakage-free) "
                "AUC is undefined and reported as NaN. Pass one group id per "
                "epoch (e.g. the character-block index).",
                RuntimeWarning,
                stacklevel=3,
            )
            return float("nan"), 0, 0

        n_groups = int(np.unique(groups).size)
        if n_groups < 2:
            warnings.warn(
                f"Only {n_groups} acquisition group(s) present; a grouped AUC "
                "requires at least 2 and is reported as NaN. Collect more than "
                "one character block before quoting a validation figure.",
                RuntimeWarning,
                stacklevel=3,
            )
            return float("nan"), 0, n_groups

        folds = int(min(cv_folds, n_groups))
        return self._pooled_auc(X, y, GroupKFold(n_splits=folds), groups), folds, n_groups

    def _shuffled_auc(self, X: np.ndarray, y: np.ndarray, cv_folds: int) -> float:
        """Shuffled stratified pooled AUC — **control only, never a result**.

        Retained solely so the optimism introduced by ignoring group structure
        can be quantified and disclosed.
        """
        min_class = int(np.min(np.bincount(y)))
        if min_class < 2:
            return float("nan")
        folds = max(2, min(cv_folds, min_class))
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        return self._pooled_auc(X, y, skf)

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
