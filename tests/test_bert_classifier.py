# encoding: utf-8
import pytest
import tempfile

import numpy as np

from wellcomeml.ml.bert_classifier import BertClassifier


@pytest.fixture
def multilabel_bert(scope='module'):
    model = BertClassifier()
    model._init_model(num_labels=4)

    return model


@pytest.mark.bert
def test_multilabel(multilabel_bert):
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

    model = multilabel_bert
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


@pytest.mark.bert
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


@pytest.mark.bert
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


@pytest.mark.bert
def test_save_load(multilabel_bert):
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

    model = multilabel_bert
    model.epochs = 1  # Only need to fit 1 epoch here really, because we're testing save
    model.fit(X, Y)

    with tempfile.TemporaryDirectory() as tmp_path:
        model.save(tmp_path)
        loaded_model = BertClassifier()
        loaded_model.load(tmp_path)

    Y_pred = loaded_model.predict(X)
    Y_prob_pred = loaded_model.predict_proba(X)
    assert Y_prob_pred.sum() >= 0
    assert Y_pred.shape == Y.shape
