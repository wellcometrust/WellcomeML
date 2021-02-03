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


def test_embed_one_sentence():
    X = ["This is a sentence"]

    for embedding in EMBEDDING_TYPES:
        vec = bert_vectorizer.BertVectorizer(sentence_embedding=embedding)
        X_embed = vec.fit_transform(X)
        assert(X_embed.shape == (1, 768))


def test_embed_two_sentences():
    X = [
        "This is a sentence",
        "This is another one"
    ]

    for embedding in EMBEDDING_TYPES:
        vec = bert_vectorizer.BertVectorizer(sentence_embedding=embedding)
        X_embed = vec.fit_transform(X)
        assert(X_embed.shape == (2, 768))


def test_embed_long_sentence():
    X = ["This is a sentence"*500]

    for embedding in EMBEDDING_TYPES:
        vec = bert_vectorizer.BertVectorizer(sentence_embedding=embedding)
        X_embed = vec.fit_transform(X)
        assert(X_embed.shape == (1, 768))


@pytest.mark.skip("Theory: Downloading scibert stalls build")
def test_embed_scibert():
    X = ["This is a sentence"]
    for embedding in EMBEDDING_TYPES:
        vec = bert_vectorizer.BertVectorizer(pretrained='scibert',
                                             sentence_embedding=embedding)
        X_embed = vec.fit_transform(X)
        assert(X_embed.shape == (1, 768))


def test_save_and_load(tmpdir):
    tmpfile = tmpdir.join('test.npy')

    X = ["This is a sentence"]
    for pretrained in ['bert', 'scibert']:
        for embedding in EMBEDDING_TYPES:
            vec = bert_vectorizer.BertVectorizer(
                pretrained=pretrained,
                sentence_embedding=embedding
            )
            X_embed = vec.fit_transform(X)

            vec.save_transformed(str(tmpfile), X_embed)

            X_loaded = vec.load_transformed(str(tmpfile))

            assert (X_loaded != X_embed).sum() == 0
