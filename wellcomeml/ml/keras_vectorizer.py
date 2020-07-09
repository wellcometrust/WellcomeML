"""
"""
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


class KerasVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vocab_size=None, sequence_length=None, oov_token="<OOV>"):
        self.vocab_size = vocab_size
        self.oov_token = oov_token
        self.sequence_length = sequence_length

    def fit(self, X, *_):
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov_token)
        self.tokenizer.fit_on_texts(X)
        return self

    def transform(self, X, *_):
        sequences = self.tokenizer.texts_to_sequences(X)
        return pad_sequences(sequences, maxlen=self.sequence_length)

    def build_embedding_matrix(self, embeddings_path):
        embeddings_index = {}
        with open(embeddings_path) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs

        emb_dim = len(coefs)
        num_words = len(self.tokenizer.word_index) + 1

        embedding_matrix = np.zeros((num_words, emb_dim))
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
