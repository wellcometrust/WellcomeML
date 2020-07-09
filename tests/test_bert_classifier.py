# encoding: utf-8
import numpy as np

from wellcomeml.ml import BertClassifier


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
    assert model.predict(X).sum() != 0
    assert model.predict(X).sum() != Y.size
    assert model.predict(X).shape == Y.shape
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
    assert model.predict(X).sum() != 0
    assert model.predict(X).sum() != np.array(Y).size
    assert model.predict(X).shape == np.array(Y).shape
    assert model.losses[0] > model.losses[-1]
