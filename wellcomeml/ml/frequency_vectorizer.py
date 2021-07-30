#!/usr/bin/env python3
# coding: utf-8

"""
A generic "frequency" vectorizer that wraps all usual transformations.
"""
import logging
import re

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from wellcomeml.utils import throw_extra_import_message
# Heavy dependencies go here
try:
    import spacy
except ImportError as e:
    throw_extra_import_message(error=e, required_module='spacy', extra='spacy')

logger = logging.getLogger(__name__)


class WellcomeTfidf(TfidfVectorizer):
    """
    Class to wrap some basic transformation and text
    vectorisation/embedding
    """

    def __init__(self, use_regex=True, use_spacy_lemmatizer=True, **kwargs):
        """

        Args:
            Any sklearn "tfidfvectorizer" arguments (min_df, etc.)

        """
        self.embedding = "tf-idf"
        self.use_regex = use_regex
        self.use_spacy_lemmatizer = use_spacy_lemmatizer

        logger.info("Initialising frequency vectorizer.")

        kwargs["stop_words"] = kwargs.get("stop_words", "english")

        super().__init__(**kwargs)

        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
        self.nlp.initialize()

    @classmethod
    def save_transformed(cls, path, X_transformed):
        """Saves transformed embedded vectors"""
        sparse.save_npz(path, X_transformed)

    @classmethod
    def load_transformed(cls, path):
        """Loads transformed embedded vectors"""
        return sparse.load_npz(path)

    def regex_transform(self, X, remove_numbers="years", *_):
        """
        Extra regular expression transformations to clean text
        Args:
            X: A list of texts (strings)
            *_:
            remove_numbers: Whether to remove years or all digits. Caveat:
            This does not only remove years, but **any number** between
            1000 and 2999.

        Returns:
            A list of texts with the applied regex transformation

        """
        if remove_numbers == "years":
            return [re.sub(r"[1-2]\d{3}", "", text) for text in X]
        elif remove_numbers == "digits":
            return [re.sub(r"\d", "", text) for text in X]
        else:
            return X

    def spacy_lemmatizer(self, X, remove_stopwords_and_punct=True):
        """
        Uses spacy pre-trained lemmatisation model to
        Args:
            X: A list of texts (strings)
            remove_stopwords_and_punct: Whether to remove stopwords,
            punctuation, pronouns

        Returns:

        """

        logger.info("Using spacy pre-trained lemmatiser.")
        if remove_stopwords_and_punct:
            return [
                [
                    token.lemma_.lower()
                    for token in doc
                    if not token.is_stop
                    and not token.is_punct
                    and token.lemma_ != "-PRON-"
                ]
                for doc in self.nlp.pipe(X)
            ]
        else:
            return [
                [token.lemma_.lower() for token in doc] for doc in self.nlp.pipe(X)
            ]

    def _pre_transform(self, X):
        if self.use_regex:
            X = self.regex_transform(X)
        if self.use_spacy_lemmatizer:
            X = self.spacy_lemmatizer(X)

        return [" ".join(text) for text in X]

    def transform(self, X):
        X = self._pre_transform(X)

        return super().transform(X)

    def fit(self, X, y=None):
        X = self._pre_transform(X)

        super().fit(X)
        return self

    def fit_transform(self, X, y=None):
        X = self._pre_transform(X)

        return super().fit_transform(X, y=y)
