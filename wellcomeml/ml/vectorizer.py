#!/usr/bin/env python3
# coding: utf-8

"""
A generic vectorizer that can fallback to tdidf or bag of words from sklearn
or embed using bert, doc2vec etc
"""

from sklearn.base import BaseEstimator, TransformerMixin
from wellcomeml.ml.frequency_vectorizer import WellcomeTfidf
from wellcomeml.ml.bert_vectorizer import BertVectorizer


class Vectorizer(BaseEstimator, TransformerMixin):
    """
    Abstract class, sklearn-compatible, that can vectorize texts using
    various models.

    """
    def __init__(self, embedding='tf-idf', **kwargs):
        """
        Args:
            embedding(str): One of `['bert', 'tf-idf']`
        """
        self.embedding = embedding

        vectorizer_dispatcher = {
            'tf-idf': WellcomeTfidf,
            'bert': BertVectorizer
        }

        if not vectorizer_dispatcher.get(embedding):
            raise ValueError(f'Model {embedding} not available')

        self.vectorizer = vectorizer_dispatcher.get(embedding)(**kwargs)

    def fit(self, X=None, *_):
        return self.vectorizer.fit(X)

    def transform(self, X, *_):
        return self.vectorizer.transform(X)

    def save(self, X_transformed, path):
        save_method = getattr(self.vectorizer.__class__, 'save', None)
        if not save_method:
            raise NotImplementedError(f'Method save not implemented for class '
                                      f'{self.vectorizer.__class__.__name__}')

        return save_method(X_transformed=X_transformed, path=path)

    def load(self, path):
        load_method = getattr(self.vectorizer.__class__, 'load', None)
        if not load_method:
            raise NotImplementedError(f'Method load not implemented for class '
                                      f'{self.vectorizer.__class__.__name__}')

        return load_method(path=path)
