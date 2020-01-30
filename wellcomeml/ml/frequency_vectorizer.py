#!/usr/bin/env python3
# coding: utf-8

"""
A generic "frequency" vectorizer that wraps all usual transformations.
"""
import re

import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from wellcomeml.logger import logger

nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser',
                                            'textcat'])


class FrequencyVectorizer(BaseEstimator, TransformerMixin):
    """
    Class to wrap some basic transformation and text
    vectorisation/embedding
    """
    def __init__(self, **kwargs):
        """

        Args:
            Any sklearn "tfidfvectorizer" arguments (min_df, etc.)

        """
        self.embedding = 'tf-idf'

        logger.info("Initialising frequency vectorizer.")
        self.vectorizer = TfidfVectorizer(stop_words='english', **kwargs)

    def regex_transform(self, X, *_, remove_numbers='years'):
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
        if remove_numbers == 'years':
            return [re.sub('[1-2]\d{3}', '', text) for text in X]
        elif remove_numbers == 'digits':
            return [re.sub('\d', '', text) for text in X]
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
        if remove_stopwords_and_punct:
            return [
                [token.lemma_.lower() for token in doc
                 if not token.is_stop and not token.is_punct and
                 token.lemma_ != "-PRON-"]
                for doc in nlp.tokenizer.pipe(X)
            ]
        else:

            return [[token.lemma_.lower() for token in
                     doc] for doc in nlp.tokenizer.pipe(X)]

    def transform(self, X, *_):
        X = self.regex_transform(X)
        X = self.spacy_lemmatizer(X)
        X = [' '.join(text) for text in X]

        return self.vectorizer.transform(X)

    def fit(self, X, *_):
        logger.info("Using spacy pre-trained lemmatiser.")
        X = self.regex_transform(X)
        X = self.spacy_lemmatizer(X)

        logger.info("Fitting vectorizer.")

        X = [' '.join(text) for text in X]
        self.vectorizer.fit(X)

        return self