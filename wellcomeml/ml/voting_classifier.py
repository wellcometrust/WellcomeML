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
    def __init__(self, multilabel=False, num_agree=None, *args, **kwargs):
        super(WellcomeVotingClassifier, self).__init__(*args, **kwargs)
        self.pretrained = self._is_pretrained()
        self.multilabel = multilabel
        self.num_agree = num_agree

    def _is_pretrained(self):
        try:
            check_is_fitted(self, "estimators")
        except NotFittedError:
            return False
        return True

    def _get_estimators(self):
        if type(self.estimators) == list:
            return [est for est in self.estimators]
        else:  # tuple with named estimators
            return [est for _, est in self.estimators]

    def _check_vectorizer(self, estimators):
        if type(estimators[0]) == tuple:
            return True
        else:
            return False

    def predict(self, X):
        if self.pretrained:
            check_is_fitted(self, "estimators")

            estimators = self._get_estimators()
            has_vectorizers = self._check_vectorizer(estimators)

            if self.voting == "soft":
                if has_vectorizers:
                    Y_probs = np.array(
                        [
                            est.predict_proba(vect.transform(X))
                            for est, vect in estimators
                        ]
                    )
                else:
                    Y_probs = np.array([est.predict_proba(X) for est in estimators])
                Y_prob = np.mean(Y_probs, axis=0)
                if self.multilabel:
                    return np.array(Y_prob > 0.5, dtype=int)
                else:
                    return np.argmax(Y_prob, axis=1)
            else:  # hard voting

                if has_vectorizers:
                    Y_preds = [
                        est.predict(vect.transform(X)) for est, vect in estimators
                    ]
                else:
                    Y_preds = [est.predict(X) for est in estimators]
                Y_preds = np.array(Y_preds)
                if self.multilabel:
                    return np.apply_along_axis(
                        lambda x: np.argmax(np.bincount(x)), axis=0, arr=Y_preds
                    )
                else:
                    # If num_agree isn't set then use majority vote
                    # TO DO: a message when if num_agree is specified, but
                    # voting=soft and/or multilabel=True then num_agree isn't used
                    # 'num_agree specified but not used - only used in binary hard voting cases'
                    if not self.num_agree:
                        # So if 4 estimators, >= 3 need to agree
                        self.num_agree = np.ceil((len(estimators) + 1) / 2)

                    return np.apply_along_axis(
                        lambda x: int(np.sum(x) >= self.num_agree), axis=0, arr=Y_preds
                    )

        else:
            return super(WellcomeVotingClassifier, self).predict(X)
