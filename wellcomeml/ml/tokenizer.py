"""
Implements Tokenizer that abstracts tokenization from hugging face,
spacy, scispacy and nltk in a common interface
"""
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders

class Tokenizer:
    def __init__(self):
        pass

    def fit(self, texts):
        self.tokenizer = Tokenizer(models.Unigram())
        self.tokenizer.normalizer = normalizers.NFKC()
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        self.decoders = decoders.ByteLevel()

        trainer = trainers.UnigramTrainer()
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

    def tokenize(self, text):
        # TODO: generalise to texts
        return self.tokenizer.encode(text).tokens

    def encode(self, text):
        # TODO: generalise to texts and two sentences
        return self.tokenizer.encode(texts)

    def save(self, tokenizer_path):
        self.tokenizer.save(tokenizer_path)

    def load(self, tokenizer_path):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
