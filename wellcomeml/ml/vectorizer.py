#!/usr/bin/env python3
# coding: utf-8

"""
A generic vectorizer that can fallback to tdidf or bag of words from sklearn
or embed using bert, doc2vec etc
"""

from sklearn.base import BaseEstimator, TransformerMixin
from wellcomeml.ml.frequency_vectorizer import FrequencyVectorizer
from wellcomeml.ml.bert_vectorizer import BertVectorizer


class Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, embedding='tf-idf', **kwargs):
        """
        Args:
            embedding(str): One of `['bert', 'tf-idf']`
        """
        self.embedding = embedding

        vectorizer_dispatcher = {
            'tf-idf': FrequencyVectorizer,
            'bert': BertVectorizer
        }

        if not vectorizer_dispatcher.get(embedding):
            raise ValueError(f'Model {embedding} not available')

        self.vectorizer = vectorizer_dispatcher.get(embedding)(**kwargs)

    def fit(self, X=None, *_):
        return self.vectorizer.fit(X)

    def transform(self, X, *_):
        return self.transform(X)
