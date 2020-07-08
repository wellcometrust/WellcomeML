# encoding: utf-8
import pytest

from wellcomeml.ml import Vectorizer


def test_bert_dispatch():
    X = ["This is a sentence"]

    text_vectorizer = Vectorizer(embedding='bert')
    X_embed = text_vectorizer.fit_transform(X)

    assert(X_embed.shape == (1, 768))


def test_tf_idf_dispatch():
    X = ['Sentence Lacking Stopwords']

    text_vectorizer = Vectorizer(embedding='tf-idf')
    X_embed = text_vectorizer.fit_transform(X)

    assert (X_embed.shape == (1, 3))


def test_wrong_model_dispatch_error():
    with pytest.raises(ValueError):
        Vectorizer(embedding='embedding_that_doesnt_exist')


def test_vectorizer_that_does_not_have_save(monkeypatch):
    X = ['This is a sentence']

    vec = Vectorizer()

    X_embed = vec.fit_transform(X)

    monkeypatch.delattr(vec.vectorizer.__class__, 'save_transformed', raising=True)
    monkeypatch.delattr(vec.vectorizer.__class__, 'load_transformed', raising=True)

    with pytest.raises(NotImplementedError):
        vec.save_transformed(path='fake_path.npy', X_transformed=X_embed)

    with pytest.raises(NotImplementedError):
        vec.load_transformed(path='fake_path.npy')
