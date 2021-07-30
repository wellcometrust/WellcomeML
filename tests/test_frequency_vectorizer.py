# encoding: utf-8
import pytest

from wellcomeml.ml.frequency_vectorizer import WellcomeTfidf


def test_tf_idf_dispatch():
    X = ['Sentence Lacking Stopwords']

    text_vectorizer = WellcomeTfidf()
    X_embed = text_vectorizer.fit_transform(X)

    assert (X_embed.shape == (1, 3))


def test_save_and_load(tmpdir):
    tmpfile = tmpdir.join('test.npz')

    X = ["This is a sentence"*100]

    vec = WellcomeTfidf()

    X_embed = vec.fit_transform(X)

    vec.save_transformed(str(tmpfile), X_embed)

    X_loaded = vec.load_transformed(str(tmpfile))

    assert (X_loaded != X_embed).sum() == 0


def test_fit_transform_and_transform():
    X = [
        "This is a sentence",
        "This is another one",
        "This is a third sentence",
        "Wellcome is a global charitable foundation",
        "We want everyone to benefit from science's potential to improve health and save lives."
    ]

    text_vectorizer = WellcomeTfidf()
    X_embed = text_vectorizer.fit_transform(X)

    X_embed_2 = text_vectorizer.transform(X)

    # Asserts that the result of transform is almost the same as fit transform
    assert (X_embed-X_embed_2).sum() == pytest.approx(0, abs=1e-6)
