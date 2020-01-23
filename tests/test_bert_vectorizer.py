# encoding: utf-8

from wellcomeml.ml import bert_vectorizer

def test_embed_one_sentence():
    X = ["This is a sentence"]

    vec = bert_vectorizer.BertVectorizer()
    X_embed = vec.fit_transform(X)
    assert(X_embed.shape == (1, 768))

def test_embed_two_sentences():
    X = [
        "This is a sentence",
        "This is another one"
    ]

    vec = bert_vectorizer.BertVectorizer()
    X_embed = vec.fit_transform(X)
    assert(X_embed.shape == (2, 768))

def test_embed_long_sentence():
    X = ["This is a sentence"*100]

    vec = bert_vectorizer.BertVectorizer()
    X_embed = vec.fit_transform(X)
    assert(X_embed.shape == (1, 768))


def test_embed_scibert():
    X = ["This is a sentence"*100]

    vec = bert_vectorizer.BertVectorizer(pretrained='scibert')
    X_embed = vec.fit_transform(X)

    # Tests if loads correctly
    assert(X_embed.shape == (1, 768))