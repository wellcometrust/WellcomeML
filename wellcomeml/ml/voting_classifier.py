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

    def predict(self, X):
        if self.pretrained:
            check_is_fitted(self, 'estimators')
            multilabel = self.multilabel
            if self.voting == "soft":
                Y_probs = np.array([clf.predict_proba(X) for clf in self.estimators])
                Y_prob = np.mean(Y_probs, axis=0)
                if multilabel:
                    return Y_prob > 0.5
                else:
                    return np.argmax(Y_prob, axis=1)
            else: # hard voting
                Y_preds = np.array([clf.predict(X) for clf in self.estimators])
                return np.apply_along_axis(
                    lambda x: np.argmax(np.bincount(x)),
                    axis=0 if self.multilabel else 1,
                    arr=Y_preds) 
        else:
            return super(WellcomeVotingClassifier, self).predict(X)
