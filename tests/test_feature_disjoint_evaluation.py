"""Synthetic holdout must not reuse feature-equivalent training examples."""

import numpy as np
import pytest

from train import _feature_disjoint_split, build_dataset, train


def test_duplicate_feature_vectors_cannot_cross_holdout_boundary():
    # Each unique feature vector occurs twice; a row-wise random split would
    # often place copies in both train and test.
    X = np.repeat(np.arange(40, dtype=float).reshape(-1, 1), 2, axis=0)
    y = np.repeat(np.array([0, 1] * 20), 2)
    train_idx, test_idx = _feature_disjoint_split(X, y, seed=7)

    train_vectors = {tuple(row) for row in X[train_idx]}
    test_vectors = {tuple(row) for row in X[test_idx]}
    assert not train_vectors & test_vectors
    assert set(y[train_idx]) == {0, 1}
    assert set(y[test_idx]) == {0, 1}


def test_real_synthetic_generator_has_disjoint_feature_groups():
    X, y = build_dataset(800, seed=7)
    assert len(np.unique(X, axis=0)) < len(X)  # Repeated observations exist.
    train_idx, test_idx = _feature_disjoint_split(X, y, seed=7)
    assert not ({tuple(row) for row in X[train_idx]}
                & {tuple(row) for row in X[test_idx]})
    assert len(train_idx) + len(test_idx) == len(X)


def test_report_discloses_split_method_and_duplicate_rows():
    _, metrics, (X_test, y_test) = train(n=800, seed=7)
    assert metrics["split"]["method"] == "feature-disjoint GroupShuffleSplit"
    assert metrics["split"]["duplicate_feature_rows"] > 0
    assert metrics["split"]["test_rows"] == len(X_test) == len(y_test)
    assert metrics["split"]["train_rows"] + metrics["split"]["test_rows"] == 800


@pytest.mark.parametrize("n", [0, 1, 7, 9, True])
def test_invalid_training_sample_counts_fail_explicitly(n):
    with pytest.raises(ValueError, match="even integer"):
        train(n=n, seed=7)
