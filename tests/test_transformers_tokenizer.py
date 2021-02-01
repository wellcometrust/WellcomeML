import tempfile

import pytest

from wellcomeml.ml import TransformersTokenizer


texts = [
    "This is a test",
    "Another sentence",
    "Don't split"
]


@pytest.fixture(scope="module")
def tokenizer():
    tokenizer = TransformersTokenizer()
    tokenizer.fit(texts)
    return tokenizer


@pytest.fixture(scope="module")
def tmp_path():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = f"{tmp_dir}/tokenizer.json"
        yield tmp_path


def test_tokenize(tokenizer):
    tokens = tokenizer.tokenize("This is a test")
    assert len(tokens) == 4
    assert type(tokens[0]) == str


def test_tokenize_batch(tokenizer):
    tokens = tokenizer.tokenize(["This is a test", "test"])
    assert len(tokens) == 2


def test_encode(tokenizer):
    token_ids = tokenizer.encode("This is a test")
    assert len(token_ids) == 4
    assert type(token_ids[0]) == int


def test_encode_batch(tokenizer):
    token_ids = tokenizer.encode(["This is a test", "test"])
    assert len(token_ids) == 2


def test_unknown_token(tokenizer):
    tokens = tokenizer.tokenize("I have not seen this before")
    assert "[UNK]" in tokens


def test_save(tokenizer, tmp_path):
    tokenizer.save(tmp_path)

    loaded_tokenizer = TransformersTokenizer()
    loaded_tokenizer.load(tmp_path)
    tokens = loaded_tokenizer.tokenize("This is a test")
    assert len(tokens) == 4


def test_bpe_model():
    tokenizer = TransformersTokenizer(model="bpe")
    tokenizer.fit(texts)
    tokens = tokenizer.tokenize("This is a test")
    assert len(tokens) == 4


def test_lowercase():
    tokenizer = TransformersTokenizer(lowercase=False)
    tokenizer.fit(texts)
    tokens = tokenizer.tokenize("This is a test")
    assert tokens[0] == "This"


def test_vocab_size():
    tokenizer = TransformersTokenizer(vocab_size=30)
    tokenizer.fit(texts)
    vocab = tokenizer.vocab
    print(vocab)
    assert len(vocab) == 30
