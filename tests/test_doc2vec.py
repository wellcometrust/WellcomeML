from wellcomeml.ml import Doc2VecVectorizer


def test_fit_transform():
    X = [
        "Wellcome trust gives grants",
        "Covid is a infeactious disease",
        "Sourdough bread is delicious",
        "Zoom is not so cool",
        "Greece is the best country",
        "Waiting for the vaccine"
    ]
    doc2vec = Doc2VecVectorizer(vector_size=8, epochs=2)
    X_vec = doc2vec.fit_transform(X)
    assert X_vec.shape == (6, 8)


def test_score():
    # It is quite difficult to construct a test where the score is reliably high
    #    so we fallback to testing that scores produced a number from 0 to 1.
    # It would be even better to test loss is decreasing but gensim does not expose loss
    X = [
        "Covid is a disease that can kill you",
        "HIV is another disease that can kill",
        "Wellcome trust funds covid and hiv research"
        "Wellcome trust is similar to NIH in the US, in that they both fund research"
        "NIH gives the most money for research every year"
    ]
    doc2vec = Doc2VecVectorizer(min_count=1, vector_size=4, negative=5, epochs=100)
    doc2vec.fit(X)
    score = doc2vec.score(X)
    assert 0 <= score <= 1
