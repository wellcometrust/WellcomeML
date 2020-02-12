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

