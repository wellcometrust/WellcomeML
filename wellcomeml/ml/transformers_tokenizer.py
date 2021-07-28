"""
Implements Tokenizer that abstracts common tokenisation
strategies used in Transformers
"""
from tokenizers.models import WordPiece, BPE
from tokenizers.normalizers import Lowercase, Sequence
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tokenizers import Tokenizer
import numpy as np

# TODO
# - generalise to two sentences
# - prepare input for transformers
# - pad and truncate to max length


class TransformersTokenizer:
    def __init__(self, lowercase=True,
                 pre_tokenizer="whitespace", model="wordpiece",
                 vocab_size=30_000, unk_token="[UNK]"):
        self.lowercase = lowercase
        self.pre_tokenizer = pre_tokenizer
        self.model = model
        self.vocab_size = vocab_size
        self.unk_token = unk_token

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['trainer']
        return attributes

    def _init_tokenizer(self):
        if self.model == "wordpiece":
            model = WordPiece(unk_token=self.unk_token)
            self.trainer = WordPieceTrainer(
                vocab_size=self.vocab_size,
                special_tokens=[self.unk_token])
        elif self.model == "bpe":
            model = BPE(unk_token=self.unk_token)
            self.trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=[self.unk_token])
        else:
            raise NotImplementedError

        normalizers = []
        if self.lowercase:
            normalizers.append(Lowercase())
        normalizers = Sequence(normalizers)

        if self.pre_tokenizer == "bytelevel":
            pre_tokenizer = ByteLevel()
        elif self.pre_tokenizer == "whitespace":
            pre_tokenizer = Whitespace()
        else:
            raise NotImplementedError

        self.tokenizer = Tokenizer(model)
        self.tokenizer.normalizer = normalizers
        self.tokenizer.pre_tokenizer = pre_tokenizer

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    def fit(self, texts):
        self._init_tokenizer()
        self.tokenizer.train_from_iterator(texts, trainer=self.trainer)

    def _yield_tokens_or_encodings(self, texts, return_type="tokens", buffer_size=1000):
        def yield_type(texts_batch, return_type):
            encodings = self.tokenizer.encode_batch(texts_batch)
            for e in encodings:
                if return_type == "tokens":
                    yield e.tokens
                elif return_type == "encodings":
                    yield e.ids
                else:
                    raise NotImplementedError

        texts_batch = []
        for text in texts:
            texts_batch.append(text)

            if len(texts_batch) >= buffer_size:
                yield from yield_type(texts_batch, return_type)
                texts_batch = []

        if texts_batch:
            yield from yield_type(texts_batch, return_type)

    def tokenize(self, text):
        if type(text) == list:
            return list(self._yield_tokens_or_encodings(text))
        elif type(text) == str:
            encoding = self.tokenizer.encode(text)
            return encoding.tokens
        else:
            raise NotImplementedError

    def encode(self, text):
        if type(text) in [list, np.ndarray]:
            return list(self._yield_tokens_or_encodings(text, return_type="encodings"))
        elif type(text) == str:
            encoding = self.tokenizer.encode(text)
            return encoding.ids
        else:
            raise NotImplementedError

    def decode(self, encoded_text):
        if not encoded_text:
            return ""
        if type(encoded_text[0]) in [list, np.ndarray]:
            return self.tokenizer.decode_batch(encoded_text)
        return self.tokenizer.decode(encoded_text)

    def save(self, tokenizer_path):
        self.tokenizer.save(tokenizer_path)

    def load(self, tokenizer_path):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        return self
