# encoding: utf-8
from wellcomeml.ml import WellcomeTfidf


def test_tf_idf_dispatch():
    X = ['Sentence Lacking Stopwords']

    text_vectorizer = WellcomeTfidf()
    X_embed = text_vectorizer.fit_transform(X)

    assert (X_embed.shape == (1, 3))
