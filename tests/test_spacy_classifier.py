# encoding: utf-8

import numpy as np

from wellcomeml.ml import SpacyClassifier


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

    model = SpacyClassifier()
    model.fit(X, Y)
    assert model.score(X, Y) > 0.3
    assert model.predict(X).shape == (5, 4)


def test_multilabel_Y_list():
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

    model = SpacyClassifier()
    model.fit(X, Y)
    assert model.score(X, Y) > 0.3
    assert model.predict(X).shape == (5, 4)


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

    model = SpacyClassifier()
    for x, y in zip(X, Y):
        model.partial_fit([x], [y])
    assert model.score(X, Y) > 0.2
    assert model.predict(X).shape == (5, 4)
