from wellcomeml.ml import Sent2VecVectorizer

def test_fit_transform():
    X = [
        "Malaria is a disease that kills people",
        "Heart problems comes first in the global burden of disease",
        "Wellcome also funds policy and culture research"
    ]
    sent2vec = Sent2VecVectorizer("sent2vec_wiki_unigrams")
    sent2vec.fit()
    X_vec = sent2vec.transform(X)
    assert X_vec.shape == (3, 600)
