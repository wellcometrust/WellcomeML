from wellcomeml.ml.bilstm import BiLSTMClassifier
from wellcomeml.ml import KerasVectorizer
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
import numpy as np


def test_vanilla():
    X = [
        "One",
        "One only",
        "Two nothing else",
        "Two and three"
    ]
    Y = np.array([0, 0, 1, 1])

    model = Pipeline([
        ('vec', KerasVectorizer()),
        ('clf', BiLSTMClassifier(nb_epochs=10))
    ])
    model.fit(X, Y)
    assert model.score(X, Y) > 0.6


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
    model = Pipeline([
        ('vec', KerasVectorizer()),
        ('clf', BiLSTMClassifier(multilabel=True))
    ])
    model.fit(X, Y)
    assert model.score(X, Y) > 0.4
    assert model.predict(X).shape == (5, 4)


def test_sparse():
    X = [
        "One and two",
        "One only",
        "Three and four, nothing else",
        "Two nothing else",
        "Two and three"
    ]
    Y = csr_matrix(np.array([
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 1, 0]
    ]))
    model = Pipeline([
        ('vec', KerasVectorizer()),
        ('clf', BiLSTMClassifier(
            multilabel=True,
            batch_size=2,
            sparse_y=True
        ))
    ])
    model.fit(X, Y)
    assert model.score(X, Y) > 0.4
    assert model.predict(X).shape == (5, 4)


def test_attention():
    X = [
        "One",
        "One only",
        "Two nothing else",
        "Two and three"
    ]
    Y = np.array([0, 0, 1, 1])

    model = Pipeline([
        ('vec', KerasVectorizer()),
        ('clf', BiLSTMClassifier(
                    nb_epochs=10,
                    attention=True,
                    attention_heads=10))
    ])
    model.fit(X, Y)
    assert model.score(X, Y) > 0.6


def test_early_stopping():
    X = [
        "One",
        "One only",
        "Two nothing else",
        "Two and three"
    ]
    Y = np.array([0, 0, 1, 1])

    model = Pipeline([
        ('vec', KerasVectorizer()),
        ('clf', BiLSTMClassifier(
                    early_stopping=True,
                    nb_epochs=10000
        ))
    ])
    # if early_stopping is not working it will take
    # a lot of time to finish running this test
    model.fit(X, Y)
    assert model.score(X, Y) > 0.6
