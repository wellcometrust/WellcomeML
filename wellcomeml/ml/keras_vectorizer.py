"""
Implements KerasTokenizer that abstracts Keras tokenisation
and encoding and KerasVectorizer that is sklearn compatible
"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import gensim.downloader as api

from os import path
import logging

from wellcomeml.ml.transformers_tokenizer import TransformersTokenizer
from wellcomeml.utils import throw_extra_import_message

try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError as e:
    throw_extra_import_message(error=e, required_module='tensorflow', extra='tensorflow')


logger = logging.getLogger(__name__)


class KerasTokenizer():
    def __init__(self, vocab_size=None, oov_token="<OOV>"):
        self.vocab_size = vocab_size
        self.oov_token = oov_token

    @property
    def vocab(self):
        return self.tokenizer.word_index

    def fit(self, texts):
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov_token)
        self.tokenizer.fit_on_texts(texts)

    def encode(self, text):
        if type(text) == str:
            return self.tokenizer.texts_to_sequences([text])[0]
        return self.tokenizer.texts_to_sequences(text)

    def decode(self, encoded_text):
        if not encoded_text:
            return ""
        if type(encoded_text[0]) == int:
            return self.tokenizer.sequences_to_texts([encoded_text])[0]
        return self.tokenizer.sequences_to_texts(encoded_text)

    def tokenize(self, text):
        if type(text) == str:
            return text.split()
        else:
            return [t.split() for t in text]


class KerasVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vocab_size=None, sequence_length=None, oov_token="<OOV>",
                 tokenizer_library="keras"):
        self.vocab_size = vocab_size
        self.oov_token = oov_token
        self.sequence_length = sequence_length
        self.tokenizer_library = tokenizer_library

    def _infer_from_data(self, X, load_buffer=1000):
        # We could look at a sample for more efficient
        max_sequence_length = 1

        def update_max_sequence_length(X_buffer, max_sequence_length, load_buffer):
            X_tokens = self.tokenizer.encode(X_buffer)
            sequence_length = max([len(x) for x in X_tokens])
            if sequence_length >= max_sequence_length:
                max_sequence_length = sequence_length
            return max_sequence_length

        X_buffer = []
        for x in X:
            X_buffer.append(x)

            if len(X_buffer) >= load_buffer:
                max_sequence_length = update_max_sequence_length(
                        X_buffer, max_sequence_length, load_buffer)
                X_buffer = []

        if X_buffer:
            max_sequence_length = update_max_sequence_length(
                X_buffer, max_sequence_length, load_buffer)

        self.sequence_length = max_sequence_length

    def fit(self, X, *_):
        if self.tokenizer_library == "keras":
            self.tokenizer = KerasTokenizer(
                vocab_size=self.vocab_size, oov_token=self.oov_token)
        elif self.tokenizer_library == "transformers":
            if self.vocab_size is None:
                self.tokenizer = TransformersTokenizer()
            else:
                self.tokenizer = TransformersTokenizer(
                    vocab_size=self.vocab_size
                )
        self.tokenizer.fit(X)
        if not self.sequence_length:
            logger.info(
                "Param sequence length not provided. Inferring from data.\
                This might take a while...")
            self._infer_from_data(X)
        return self

    def transform(self, X, *_):
        sequences = self.tokenizer.encode(X)
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

        num_words = len(self.tokenizer.vocab) + 1

        embedding_matrix = np.zeros((num_words, emb_dim))
        for word, i in self.tokenizer.vocab.items():
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
