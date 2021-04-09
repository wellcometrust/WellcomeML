# encoding: utf-8
import pytest

from wellcomeml.ml import bert_vectorizer

EMBEDDING_TYPES = [
    "mean_second_to_last",
    "mean_last",
    "sum_last",
    "mean_last_four",
    "pooler"
]


@pytest.fixture
def vec():
    vectorizer = bert_vectorizer.BertVectorizer()

    vectorizer.fit()
    return vectorizer


def test_fit_transform_works(vec):
    X = ["This is a sentence"]

    assert vec.fit_transform(X).shape == (1, 768)


def test_embed_two_sentences(vec):
    X = [
        "This is a sentence",
        "This is another one"
    ]

    for embedding in EMBEDDING_TYPES:
        vec.sentence_embedding = embedding
        X_embed = vec.transform(X, verbose=False)
        assert X_embed.shape == (2, 768)


def test_embed_long_sentence(vec):
    X = ["This is a sentence"*500]

    for embedding in EMBEDDING_TYPES:
        vec.sentence_embedding = embedding
        X_embed = vec.transform(X, verbose=False)
        assert X_embed.shape == (1, 768)


def test_embed_scibert():
    X = ["This is a sentence"]
    vec = bert_vectorizer.BertVectorizer(pretrained='scibert')
    vec.fit()

    for embedding in EMBEDDING_TYPES:
        vec.sentence_embedding = embedding
        X_embed = vec.transform(X, verbose=False)
        assert X_embed.shape == (1, 768)


@pytest.mark.skip("Reason: Build killed or stalls. Issue #200")
def test_save_and_load(tmpdir):
    tmpfile = tmpdir.join('test.npy')

    X = ["This is a sentence"]
    for pretrained in ['bert', 'scibert']:
        for embedding in EMBEDDING_TYPES:
            vec = bert_vectorizer.BertVectorizer(
                pretrained=pretrained,
                sentence_embedding=embedding
            )
            X_embed = vec.fit_transform(X, verbose=False)

            vec.save_transformed(str(tmpfile), X_embed)

            X_loaded = vec.load_transformed(str(tmpfile))

            assert (X_loaded != X_embed).sum() == 0
