import tempfile

from wellcomeml.ml import CNNClassifier, KerasVectorizer
from sklearn.pipeline import Pipeline
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
        ('clf', CNNClassifier())
    ])
    model.fit(X, Y)
    assert model.score(X, Y) > 0.6


def test_save_load():
    X = [
        "One",
        "One only",
        "Two nothing else",
        "Two and three"
    ]
    Y = np.array([0, 0, 1, 1])

    vec = KerasVectorizer()
    X_vec = vec.fit_transform(X)

    model = CNNClassifier()
    model.fit(X_vec, Y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save(tmp_dir)
        loaded_model = CNNClassifier()
        loaded_model.load(tmp_dir)
        assert hasattr(loaded_model, 'model')
        assert loaded_model.score(X_vec, Y) > 0.6


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
        ('clf', CNNClassifier(multilabel=True))
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
        ('clf', CNNClassifier(
                    attention=True,
                    attention_heads=10))
    ])
    model.fit(X, Y)
    assert model.score(X, Y) > 0.6
