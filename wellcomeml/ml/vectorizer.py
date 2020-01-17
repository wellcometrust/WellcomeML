#!/usr/bin/env python3
# coding: utf-8

"""
A generic vectorizer that can fallback to tdidf or bag of words from sklearn
or embed using bert, doc2vec etc
"""

import numpy as np

import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel, BertTokenizer


class Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, embedding='vectorizer', **kwargs):
        """
        Args:
            embedding(str): One of `['vectorizer', 'tfidf']`
        """
        self.embedding = embedding

        if embedding == 'tfidf':
            self.vectorizer = TfidfVectorizer(**kwargs)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model.eval()

    def bert_embedding(self, x):
        """
        Args:
            x(str):?
        """
        tokenized_x = self.tokenizer.tokenize("[CLS] " + x + " [SEP]")
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_x)

        # Max sequence length is 512

        if len(indexed_tokens) > 512:
            # Also we want to maintain the SEP token last
            indexed_tokens = indexed_tokens[:511] + [indexed_tokens[-1]]
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.zeros(tokens_tensor.shape, dtype=torch.long)
        with torch.no_grad():
            output = self.model(tokens_tensor, token_type_ids=segments_tensor)
        last_layer = output[2][-1]
        second_to_last_layer = output[2][-2]

        embedded_x = second_to_last_layer.mean(axis=1)

        return embedded_x.numpy().flatten()

    def transform(self, X, *_):
        if self.embedding == 'tfidf':
            return self.vectorizer.transform(X)
        else:
            return np.array([self.bert_embedding(x) for x in X])


    def fit(self, X, *_):
        if self.embedding == 'tfidf':
            self.vectorizer.fit(X)
        else:
            pass # BERT is already trained

        return self
