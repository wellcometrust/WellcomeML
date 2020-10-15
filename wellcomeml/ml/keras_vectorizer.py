"""
"""
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import gensim.downloader as api

from os import path


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

    def build_embedding_matrix(self, embeddings_name_or_path=None):
        """
        Builds an embedding matrix from either a local embeddings path
        or a gensim pre-trained word vector path

        Args:
            embeddings_name_or_path:
                Can be either:
                - A local directory to word embeddings
                - The name of a GenSim pre-trained word vector model
                    e.g. 'glove-twitter-25', for the complete list:
                    https://github.com/RaRe-Technologies/gensim-data#models

        Returns:
            An embedding matrix

        """
        local_embeddings = False
        if path.isfile(embeddings_name_or_path):
            try:
                embeddings_index = {}
                with open(embeddings_name_or_path) as f:
                    for line in f:
                        word, coefs = line.split(maxsplit=1)
                        coefs = np.fromstring(coefs, "f", sep=" ")
                        embeddings_index[word] = coefs
                    emb_dim = len(coefs)
                local_embeddings = True
            except TypeError:
                raise TypeError("Incorrect local embeddings path")
        elif embeddings_name_or_path:
            try:
                embeddings_index = api.load(embeddings_name_or_path)
                emb_dim = embeddings_index.vector_size
            except ValueError:
                raise ValueError(
                    "Incorrect GenSim word vector model name, try e.g. 'glove-twitter-25'"
                )
        else:
            raise TypeError("No local or GenSim word embeddings given")

        num_words = len(self.tokenizer.word_index) + 1

        embedding_matrix = np.zeros((num_words, emb_dim))
        for word, i in self.tokenizer.word_index.items():
            if local_embeddings:
                embedding_vector = embeddings_index.get(word)
            else:
                # get_vector will error if the word isn't in the vocab
                try:
                    embedding_vector = embeddings_index.get_vector(word)
                except KeyError:
                    embedding_vector = None
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
