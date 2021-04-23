import tempfile

from wellcomeml.ml.bilstm import BiLSTMClassifier
from wellcomeml.ml.keras_vectorizer import KerasVectorizer
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

    model = BiLSTMClassifier()
    model.fit(X_vec, Y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save(tmp_dir)
        loaded_model = BiLSTMClassifier()
        loaded_model.load(tmp_dir)
        assert hasattr(loaded_model, 'model')
        assert loaded_model.score(X_vec, Y) > 0.6


def test_save_load_attention():
    X = [
        "One",
        "One only",
        "Two nothing else",
        "Two and three"
    ]
    Y = np.array([0, 0, 1, 1])

    vec = KerasVectorizer()
    X_vec = vec.fit_transform(X)

    model = BiLSTMClassifier(attention=True)
    model.fit(X_vec, Y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save(tmp_dir)
        loaded_model = BiLSTMClassifier()
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


def test_predict_proba():
    X = [
        "One",
        "One only",
        "Two nothing else",
        "Two and three"
    ]
    Y = np.array([0, 0, 1, 1])

    model = Pipeline([
        ('vec', KerasVectorizer()),
        ('clf', BiLSTMClassifier())
    ])
    model.fit(X, Y)
    Y_pred_prob = model.predict_proba(X)
    assert sum(Y_pred_prob >= 0) == Y.shape[0]
    assert sum(Y_pred_prob <= 1) == Y.shape[0]


def test_threshold():
    X = [
        "One",
        "One only",
        "Two nothing else",
        "Two and three"
    ]
    Y = np.array([0, 0, 1, 1])

    model = Pipeline([
        ('vec', KerasVectorizer()),
        ('clf', BiLSTMClassifier(threshold=0.1))
    ])
    model.fit(X, Y)
    Y_pred_expected = model.predict_proba(X) > 0.1
    Y_pred = model.predict(X)
    assert np.array_equal(Y_pred_expected, Y_pred)
