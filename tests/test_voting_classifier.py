import numpy as np

from wellcomeml.ml import WellcomeVotingClassifier


class MockEstimator():
    def __init__(self, Y_prob, multilabel=False):
        self.fittted_ = True
        self.Y_prob = Y_prob
        self.multilabel = multilabel
        self.Y = self.get_Y()

    def get_Y(self):
        if self.multilabel:
            Y = self.Y_prob > 0.5
            return Y.astype(int)
        else:
            Y = np.argmax(self.Y_prob, axis=1)
            return Y

    def fit(self, X, Y):
        pass

    def predict(self, X):
        return self.Y

    def predict_proba(self, X):
        return self.Y_prob


def test_multilabel():
    Y1_prob = np.array([
        [0.9, 0.5],
        [0.2, 0.8],
        [0.3, 0.9]
    ])
    Y2_prob = np.array([
        [0.7, 0.6],
        [0.6, 0.5],
        [0.4, 0.3]
    ])
    Y_expected = np.array([
        [1, 1],
        [0, 1],
        [0, 1]
    ])
    est1 = MockEstimator(Y1_prob, multilabel=True)
    est2 = MockEstimator(Y2_prob, multilabel=True)

    voting_classifier = WellcomeVotingClassifier(
        estimators=[est1, est2], voting="soft", multilabel=True
    )
    X = ["mock", "data", "not used"]
    Y = voting_classifier.predict(X)
    assert np.array_equal(Y, Y_expected)


def test_binary():
    Y1_prob = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.3, 0.7]
    ])
    Y2_prob = np.array([
        [0.7, 0.3],
        [0.9, 0.1],
        [0.4, 0.6]
    ])
    Y_expected = np.array([
        0,
        0,
        1
    ])
    est1 = MockEstimator(Y1_prob)
    est2 = MockEstimator(Y2_prob)

    voting_classifier = WellcomeVotingClassifier(
        estimators=[est1, est2], voting="soft"
    )
    X = ["mock", "data", "not used"]
    Y = voting_classifier.predict(X)
    assert np.array_equal(Y, Y_expected)


def test_multiclass():
    Y1_prob = np.array([
        [0.6, 0.1, 0.1, 0.2],
        [0.2, 0.3, 0.1, 0.4],
        [0.3, 0.1, 0.4, 0.2]
    ])
    Y2_prob = np.array([
        [0.5, 0.3, 0.2, 0.0],
        [0.1, 0.6, 0.1, 0.2],
        [0.1, 0.3, 0.1, 0.5]
    ])
    Y_expected = np.array([
        0,
        1,
        3
    ])
    est1 = MockEstimator(Y1_prob)
    est2 = MockEstimator(Y2_prob)

    voting_classifier = WellcomeVotingClassifier(
        estimators=[est1, est2], voting="soft"
    )
    X = ["mock", "data", "not used"]
    Y = voting_classifier.predict(X)
    assert np.array_equal(Y, Y_expected)


def test_hard_voting():
    Y1_prob = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.3, 0.7]
    ])
    Y2_prob = np.array([
        [0.7, 0.3],
        [0.9, 0.1],
        [0.4, 0.6]
    ])
    Y3_prob = np.array([
        [0.3, 0.7],
        [0.8, 0.2],
        [0.4, 0.6]
    ])
    Y_expected = np.array([
        0,
        0,
        1
    ])
    est1 = MockEstimator(Y1_prob)
    est2 = MockEstimator(Y2_prob)
    est3 = MockEstimator(Y3_prob)

    voting_classifier = WellcomeVotingClassifier(
        estimators=[est1, est2, est3]
    )
    X = ["mock", "data", "not used"]
    Y = voting_classifier.predict(X)
    assert np.array_equal(Y, Y_expected)
