"""Unit tests for :mod:`classifier`, focused on the validation policy.

The central property under test is that performance is estimated with *grouped*
cross-validation, that a grouped estimate is refused rather than faked when no
group structure is available, and that an implausibly high grouped score is
flagged as a probable leakage defect instead of being reported as skill.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from classifier import (
    DEFAULT_LEAKAGE_GATE_AUC,
    NON_TARGET,
    TARGET,
    P300Classifier,
    decode_character_scores,
)


# --------------------------------------------------------------------------- #
# Synthetic data helpers
# --------------------------------------------------------------------------- #
def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _blocked_dataset(
    n_groups: int = 6,
    per_group: int = 24,
    target_fraction: float = 0.25,
    effect: float = 0.35,
    block_effect: float = 0.0,
    n_features: int = 12,
    seed: int = 0,
):
    """Build a blocked, labelled dataset.

    Args:
        effect: Amplitude of the genuine target signal along a single direction
            shared by every block. This is the part that generalises.
        block_effect: Amplitude of a target signal along a direction that is
            *unique to each block*. This is the leakage mechanism: a model that
            has seen other epochs from the same block can exploit it, and a
            model evaluated on a held-out block cannot. Set it above ``effect``
            to make ignoring group structure measurably optimistic.

    Returns:
        ``(X, y, groups)``.
    """
    rng = np.random.default_rng(seed)
    shared_dir = _unit(rng.normal(size=n_features))
    X_parts, y_parts, g_parts = [], [], []
    n_targets = max(1, int(round(per_group * target_fraction)))
    for g in range(n_groups):
        y_g = np.array([TARGET] * n_targets + [NON_TARGET] * (per_group - n_targets))
        rng.shuffle(y_g)
        block_dir = _unit(rng.normal(size=n_features))
        direction = effect * shared_dir + block_effect * block_dir
        X_g = rng.normal(0.0, 1.0, size=(per_group, n_features))
        X_g += y_g[:, None] * direction[None, :]
        X_parts.append(X_g)
        y_parts.append(y_g)
        g_parts.append(np.full(per_group, g, dtype=int))
    return np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(g_parts)


# --------------------------------------------------------------------------- #
# Grouped cross-validation
# --------------------------------------------------------------------------- #
def test_grouped_auc_is_finite_and_folds_are_reported() -> None:
    X, y, groups = _blocked_dataset(n_groups=6)
    clf = P300Classifier()
    report = clf.fit(X, y, groups=groups, cv_folds=5)
    assert np.isfinite(report.grouped_auc)
    assert 0.0 <= report.grouped_auc <= 1.0
    assert report.n_groups == 6
    assert report.grouped_folds == 5          # min(cv_folds=5, n_groups=6)


def test_fold_count_is_capped_by_group_count() -> None:
    X, y, groups = _blocked_dataset(n_groups=3)
    report = P300Classifier().fit(X, y, groups=groups, cv_folds=5)
    assert report.n_groups == 3
    assert report.grouped_folds == 3          # min(5, 3)


def test_missing_groups_yields_nan_and_warns() -> None:
    """Without groups the grouped estimate is undefined. It must be NaN and
    loud — never silently backfilled with the shuffled number."""
    X, y, _ = _blocked_dataset(n_groups=4)
    with pytest.warns(RuntimeWarning, match="No acquisition groups"):
        report = P300Classifier().fit(X, y, groups=None)
    assert np.isnan(report.grouped_auc)
    assert report.n_groups == 0
    assert report.grouped_folds == 0
    # The shuffled control is still computed, but it is not the grouped result.
    assert np.isfinite(report.shuffled_auc)


def test_single_group_yields_nan_and_warns() -> None:
    X, y, groups = _blocked_dataset(n_groups=1, per_group=40)
    with pytest.warns(RuntimeWarning, match="Only 1 acquisition group"):
        report = P300Classifier().fit(X, y, groups=groups)
    assert np.isnan(report.grouped_auc)
    assert report.n_groups == 1


def test_shuffled_control_is_optimistic_when_effect_is_block_specific() -> None:
    """When most of the discriminative signal is block-specific, shuffling
    epochs across the train/test boundary lets the model learn a pattern that
    does not generalise to a new block. The optimism term must expose that gap
    — this is precisely the mechanism that produced the discarded 94% figure."""
    X, y, groups = _blocked_dataset(
        n_groups=8, per_group=30, effect=0.1, block_effect=2.5, seed=3
    )
    report = P300Classifier().fit(X, y, groups=groups, cv_folds=5)
    assert np.isfinite(report.grouped_auc)
    assert np.isfinite(report.shuffled_auc)
    assert report.optimism == pytest.approx(
        report.shuffled_auc - report.grouped_auc
    )
    assert report.shuffled_auc > report.grouped_auc


def test_optimism_matches_definition() -> None:
    X, y, groups = _blocked_dataset(n_groups=5)
    report = P300Classifier().fit(X, y, groups=groups)
    assert report.optimism == pytest.approx(
        report.shuffled_auc - report.grouped_auc
    )


# --------------------------------------------------------------------------- #
# Leakage gate
# --------------------------------------------------------------------------- #
def test_leakage_gate_trips_on_implausibly_high_grouped_auc() -> None:
    """A trivially separable problem produces AUC ~1.0. On a 3-channel montage
    that is a bug, and the report must say so."""
    X, y, groups = _blocked_dataset(n_groups=5, effect=25.0, seed=1)
    with pytest.warns(RuntimeWarning, match="leakage gate"):
        report = P300Classifier().fit(X, y, groups=groups)
    assert report.grouped_auc > DEFAULT_LEAKAGE_GATE_AUC
    assert report.leakage_flag is True
    assert "LEAKAGE GATE TRIPPED" in str(report)


def test_leakage_gate_not_tripped_on_realistic_performance() -> None:
    X, y, groups = _blocked_dataset(n_groups=6, effect=0.35, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = P300Classifier().fit(X, y, groups=groups)
    assert report.grouped_auc < DEFAULT_LEAKAGE_GATE_AUC
    assert report.leakage_flag is False


def test_leakage_gate_threshold_is_configurable() -> None:
    X, y, groups = _blocked_dataset(n_groups=6, effect=0.8, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        strict = P300Classifier().fit(X, y, groups=groups, leakage_gate_auc=0.10)
    assert strict.leakage_flag is True


# --------------------------------------------------------------------------- #
# Input contract
# --------------------------------------------------------------------------- #
def test_fit_rejects_single_class() -> None:
    X = np.random.default_rng(0).normal(size=(20, 5))
    y = np.zeros(20, dtype=int)
    with pytest.raises(ValueError, match="single class"):
        P300Classifier().fit(X, y, groups=np.arange(20) // 5)


def test_fit_rejects_mismatched_group_length() -> None:
    X, y, _ = _blocked_dataset(n_groups=4)
    with pytest.raises(ValueError, match="one group id per epoch"):
        P300Classifier().fit(X, y, groups=np.arange(len(y) - 1))


def test_fit_rejects_empty_design_matrix() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        P300Classifier().fit(np.empty((0, 5)), np.empty((0,), dtype=int))


# --------------------------------------------------------------------------- #
# Inference and persistence
# --------------------------------------------------------------------------- #
def test_decision_scores_rank_targets_above_non_targets() -> None:
    X, y, groups = _blocked_dataset(n_groups=5, effect=1.5, seed=5)
    clf = P300Classifier()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y, groups=groups)
    scores = clf.decision_scores(X)
    assert scores.shape == (X.shape[0],)
    assert scores[y == TARGET].mean() > scores[y == NON_TARGET].mean()


def test_unfitted_classifier_refuses_to_score() -> None:
    with pytest.raises(RuntimeError):
        P300Classifier().decision_scores(np.zeros((3, 5)))


def test_save_and_load_roundtrip(tmp_path) -> None:
    X, y, groups = _blocked_dataset(n_groups=4, effect=1.0, seed=6)
    clf = P300Classifier()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y, groups=groups)
    path = str(tmp_path / "model.joblib")
    clf.save(path)
    restored = P300Classifier.load(path)
    np.testing.assert_allclose(
        restored.decision_scores(X), clf.decision_scores(X)
    )


def test_decode_character_scores_picks_argmax() -> None:
    row = np.array([0.1, 0.9, 0.2])
    col = np.array([0.4, 0.2, 0.8, 0.1])
    assert decode_character_scores(row, col) == (1, 2)
