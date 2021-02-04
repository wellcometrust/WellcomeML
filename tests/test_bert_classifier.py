# encoding: utf-8
import pytest

import numpy as np

from wellcomeml.ml import BertClassifier


@pytest.mark.skip("Theory: Too much memory")
def test_multilabel():
    X = [
        "One and two",
        "One only",
        "Three and four, nothing else",
        "Two nothing else",
        "Two and three"
    ]
    Y = np.array([
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 1, 0]
    ])

    model = BertClassifier()
    model.fit(X, Y)
    Y_pred = model.predict(X)
    Y_prob_pred = model.predict_proba(X)
    assert Y_pred.sum() != 0
    assert Y_pred.sum() != Y.size
    assert Y_prob_pred.max() <= 1
    assert Y_prob_pred.min() >= 0
    assert Y_pred.shape == Y.shape
    assert Y_prob_pred.shape == Y.shape
    assert model.losses[0] > model.losses[-1]


@pytest.mark.skip("Theory: Too much memory")
def test_multiclass():
    X = [
        "One oh yes",
        "Two noo",
        "Three ok",
        "one fantastic",
        "two bad"
    ]
    Y = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])

    model = BertClassifier(multilabel=False)
    model.fit(X, Y)
    Y_pred = model.predict(X)
    Y_prob_pred = model.predict_proba(X)
    assert Y_pred.sum() != 0
    assert Y_pred.sum() != Y.size
    assert Y_prob_pred.max() <= 1
    assert Y_prob_pred.min() >= 0
    assert Y_pred.shape == Y.shape
    assert Y_prob_pred.shape == Y.shape
    assert model.losses[0] > model.losses[-1]


@pytest.mark.skip("Theory: Too much memory")
def test_scibert():
    X = [
        "One and two",
        "One only",
        "Three and four, nothing else",
        "Two nothing else",
        "Two and three"
    ]
    Y = np.array([
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 1, 0]
    ])

    model = BertClassifier(pretrained="scibert")
    model.fit(X, Y)
    Y_pred = model.predict(X)
    Y_prob_pred = model.predict_proba(X)
    assert Y_pred.sum() != 0
    assert Y_pred.sum() != Y.size
    assert Y_prob_pred.max() <= 1
    assert Y_prob_pred.min() >= 0
    assert Y_pred.shape == Y.shape
    assert Y_prob_pred.shape == Y.shape
    assert model.losses[0] > model.losses[-1]
