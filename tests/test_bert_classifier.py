# encoding: utf-8
import numpy as np

from wellcomeml.ml.bert_classifier import BertClassifier


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


def test_partial_fit():
    X = [
        "One and two",
        "One only",
        "Three and four, nothing else",
        "Two nothing else",
        "Two and three"
    ]
    Y = [
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 1, 0]
    ]

    model = BertClassifier()
    for epoch in range(5):
        for x, y in zip(X, Y):
            model.partial_fit([x], np.array([y]))
    Y_pred = model.predict(X)
    Y_prob_pred = model.predict_proba(X)
    assert Y_pred.sum() != 0
    assert Y_prob_pred.sum() != np.array(Y).size
    assert Y_prob_pred.max() <= 1
    assert Y_prob_pred.min() >= 0
    assert Y_pred.shape == np.array(Y).shape
    assert Y_prob_pred.shape == np.array(Y).shape
    assert model.losses[0] > model.losses[-1]


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
