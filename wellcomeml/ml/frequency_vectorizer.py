#!/usr/bin/env python3
# coding: utf-8

"""
A generic "frequency" vectorizer that wraps all usual transformations.
"""
import re

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from wellcomeml.logger import logger


class WellcomeTfidf(TfidfVectorizer):
    """
    Class to wrap some basic transformation and text
    vectorisation/embedding
    """

    def __init__(self, **kwargs):
        """

        Args:
            Any sklearn "tfidfvectorizer" arguments (min_df, etc.)

        """
        self.embedding = "tf-idf"

        logger.info("Initialising frequency vectorizer.")

        kwargs["stop_words"] = kwargs.get("stop_words", "english")

        super().__init__(**kwargs)

    @classmethod
    def save_transformed(cls, path, X_transformed):
        """Saves transformed embedded vectors"""
        # Numpy will save the transformed sparse array
        # into a 0-dimensional array, preserving sparsity
        np.save(path, X_transformed)

    @classmethod
    def load_transformed(cls, path):
        """Loads transformed embedded vectors"""

        # Loads the 0-dimensional array and returns its content
        X_transformed = np.load(path, allow_pickle=True)

        return X_transformed[()]

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

        try:
            nlp = spacy.load(
                "en_core_web_sm", disable=["ner", "tagger", "parser", "textcat"]
            )
        except IOError:
            from wellcomeml.__main__ import download

            download("models")
            # pkg_resources need to be reloaded to pick up the newly installed models
            import pkg_resources
            import imp

            imp.reload(pkg_resources)
            nlp = spacy.load(
                "en_core_web_sm", disable=["ner", "tagger", "parser", "textcat"]
            )

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
                for doc in nlp.tokenizer.pipe(X)
            ]
        else:
            return [
                [token.lemma_.lower() for token in doc] for doc in nlp.tokenizer.pipe(X)
            ]

    def transform(self, X, regex=True, spacy_lemmatizer=True, *_):
        if regex:
            X = self.regex_transform(X)
        if spacy_lemmatizer:
            X = self.spacy_lemmatizer(X)

        X = [" ".join(text) for text in X]

        return super().transform(X)

    def fit(self, X, regex=True, spacy_lemmatizer=True, *_):
        if regex:
            X = self.regex_transform(X)
        if spacy_lemmatizer:
            X = self.spacy_lemmatizer(X)

        logger.info("Fitting vectorizer.")

        X = [" ".join(text) for text in X]
        super().fit(X)

        return self
