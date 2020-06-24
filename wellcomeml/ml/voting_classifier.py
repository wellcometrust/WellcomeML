"""
Voting classifier with ability to accept pretrained estimators
and return results for multilabel Y

Some of the code is taken from https://gist.github.com/tomquisel/a421235422fdf6b51ec2ccc5e3dee1b4
"""
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import VotingClassifier
from sklearn.exceptions import NotFittedError
import numpy as np


class WellcomeVotingClassifier(VotingClassifier):
    def __init__(self, *args, multilabel=False, **kwargs):
        super(WellcomeVotingClassifier, self).__init__(*args, **kwargs)
        self.pretrained = self._is_pretrained()
        self.multilabel = multilabel

    def _is_pretrained(self):
        try:
            check_is_fitted(self, 'estimators')
        except NotFittedError:
            return False
        return True

    def _is_multilabel(self, X):
        X_sample = X[:5]
        estimators = self._get_estimators()
        Y = estimators[0].predict(X_sample)
        return Y.ndim > 1

    def _get_estimators(self):
        if type(self.estimators) == list:
            return [est for est in self.estimators]
        else: # tuple with named estimators
            return [est for _, est in self.estimators]

    def predict(self, X):
        if self.pretrained:
            check_is_fitted(self, 'estimators')

            multilabel = self._is_multilabel(X)
            estimators = self._get_estimators()

            if self.voting == "soft":
                Y_probs = np.array([est.predict_proba(X) for est in estimators])
                Y_prob = np.mean(Y_probs, axis=0)
                if multilabel:
                    return np.array(Y_prob > 0.5, dtype=int)
                else:
                    return np.argmax(Y_prob, axis=1)
            else: # hard voting
                Y_preds = np.array([est.predict(X) for est in estimators])
                return np.apply_along_axis(
                    lambda x: np.argmax(np.bincount(x)),
                    axis=0 if multilabel else 1,
                    arr=Y_preds)
        else:
            return super(WellcomeVotingClassifier, self).predict(X)
