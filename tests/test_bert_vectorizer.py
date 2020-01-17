# encoding: utf-8

from wellcomeml.ml import BertVectorizer

def test_embed_one_sentence():
    X = ["This is a sentence"]

    vec = BertVectorizer()
    X_embed = vec.fit_transform(X)
    assert(X_embed.shape == (1, 768))

def test_embed_two_sentences():
    X = [
        "This is a sentence",
        "This is another one"
    ]

    vec = BertVectorizer()
    X_embed = vec.fit_transform(X)
    assert(X_embed.shape == (2, 768))

def test_embed_long_sentence():
    X = ["This is a sentence"*100]

    vec = BertVectorizer()
    X_embed = vec.fit_transform(X)
    assert(X_embed.shape == (1, 768))

def test_embed_scibert():
    X = ["This is a sentence"]

    #vec = BertVectorizer(pretrained='scibert')
    #X_embed = vec.fit_transform(X)

    # TODO: This requires scibert downloaded so ask what is
    # the best way to do that.
    # assert(X_embed.shape == (1, 768)) 
