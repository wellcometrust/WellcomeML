#!/usr/bin/env python3
# coding: utf-8

"""
A generic vectorizer that can fallback to tdidf or bag of words from sklearn
or embed using bert, doc2vec etc
"""

from sklearn.base import BaseEstimator, TransformerMixin
from wellcomeml.ml.frequency_vectorizer import WellcomeTfidf
from wellcomeml.ml.bert_vectorizer import BertVectorizer
from wellcomeml.ml.keras_vectorizer import KerasVectorizer
from wellcomeml.ml.doc2vec_vectorizer import Doc2VecVectorizer


class Vectorizer(BaseEstimator, TransformerMixin):
    """
    Abstract class, sklearn-compatible, that can vectorize texts using
    various models.

    """

    def __init__(self, embedding="tf-idf", cache_transformed=False, **kwargs):
        """
        Args:
            embedding(str): One of `['bert', 'tf-idf']`
            cache_transformed(bool): Caches the last transformed vector X (
            useful if performing Grid-search as part of a pipeline)
        """
        self.embedding = embedding
        self.cache_transformed = cache_transformed

        vectorizer_dispatcher = {
            "tf-idf": WellcomeTfidf,
            "bert": BertVectorizer,
            "keras": KerasVectorizer,
            "doc2vec": Doc2VecVectorizer,
        }

        if not vectorizer_dispatcher.get(embedding):
            raise ValueError(f"Model {embedding} not available")

        self.vectorizer = vectorizer_dispatcher.get(embedding)(**kwargs)

    def fit(self, X=None, *_):
        return self.vectorizer.fit(X)

    def transform(self, X, *_):
        X_transformed = self.vectorizer.transform(X)

        if self.cache_transformed:
            self.X_transformed = X_transformed

        return X_transformed

    def fit_transform(self, X, y=None, *_):
        # Slightly modified fit_transform so it can work with the
        # cache_transformed
        self.fit(X)
        return self.transform(X)

    def save_transformed(self, path, X_transformed):
        """
        Saves transformed vector X_transformed vector, using the corresponding
        save_transformed method for the specific vectorizer.

        Args:
            path: A path to the embedding file
            X_transformed: A transformed vector (as output by using the
            .transform method)

        """
        save_method = getattr(self.vectorizer.__class__, "save_transformed", None)
        if not save_method:
            raise NotImplementedError(
                f"Method save_transformed not implemented"
                f" for class "
                f"{self.vectorizer.__class__.__name__}"
            )

        return save_method(path=path, X_transformed=X_transformed)

    def load_transformed(self, path):
        """
        Loads transformed vector X_transformed vector, using the corresponding
        load method for the specific vectorizer.

        Args:
            path: A path to the file containing embedded vectors

        Returns:
            X_transformed (array), like the one returned by the the
            fit_transform function.
        """
        load_method = getattr(self.vectorizer.__class__, "load_transformed", None)
        if not load_method:
            raise NotImplementedError(
                f"Method load_transformed not implemented"
                f" for class "
                f"{self.vectorizer.__class__.__name__}"
            )

        return load_method(path=path)
