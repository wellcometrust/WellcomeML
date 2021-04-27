"""
Voting classifier with ability to accept pretrained estimators
and return results for multilabel Y

Some of the code is taken from https://gist.github.com/tomquisel/a421235422fdf6b51ec2ccc5e3dee1b4

voting = "soft":
    multilabel = True:
       The predicted classes will be any classes where the mean probability from all the
    estimators is > 0.5.
    multilabel = False:
        The predicted class will be class with the largest mean probability from all the
        estimators.

voting = "hard":
    The predicted class(es) will be any class(es) where the majority (or >= num_agree) of
    estimators agree. Majority in even cases is 2/2 or >=3/4 (i.e. not 2/4).

    If there are multiple classes with the majority (a tie) then the first one in
    numerical order will be chosen.

    In the multiclass case, if there are multiple classes with >= num_agree, then the
    predicted class will be the first class in a numerically sorted list of the tying classes.
    e.g.
        If there is 1 vote for class 0, 2 votes for class 1 and 2 votes for class 2,
        then class 1 will be predicted.

    In the binary case, if there is a tie then class 0 is predicted.
    e.g.
        If there are 2 votes for class 0 and 2 votes for class 1, then class 0
         will be predicted.

"""
import logging

from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import VotingClassifier
from sklearn.exceptions import NotFittedError
import numpy as np

logger = logging.getLogger(__name__)


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

    def predict(self, X):
        if self.pretrained:
            check_is_fitted(self, "estimators")

            estimators = self._get_estimators()

            if self.voting == "soft":
                if self.num_agree:
                    logger.warning("num_agree specified but not used in soft voting")
                Y_probs = np.array([est.predict_proba(X) for est in estimators])
                Y_prob = np.mean(Y_probs, axis=0)
                if self.multilabel:
                    return np.array(Y_prob > 0.5, dtype=int)
                else:
                    return np.argmax(Y_prob, axis=1)
            else:  # hard voting

                # If num_agree isn't set then use majority vote
                if not self.num_agree:
                    # So if 4 estimators, >= 3 need to agree
                    self.num_agree = np.ceil((len(estimators) + 1) / 2)

                Y_preds = [est.predict(X) for est in estimators]
                Y_preds = np.array(Y_preds)
                if self.multilabel:
                    return np.array(Y_preds.sum(axis=0) >= self.num_agree, dtype='int32')
                else:
                    votes = np.apply_along_axis(lambda x: max(np.bincount(x)), axis=0, arr=Y_preds)
                    max_class = np.apply_along_axis(
                        lambda x: np.argmax(np.bincount(x)), axis=0, arr=Y_preds
                    )
                    # If no maximum over the threshold, then pick the first from an ordered list
                    # of the other options, e.g. if 5,2,3 were voted on pick 2 (not 0)
                    options = np.sort(np.transpose(Y_preds))
                    return [m if v >= self.num_agree else options[i][0]
                            for i, (m, v) in enumerate(zip(max_class, votes))]

        else:
            return super(WellcomeVotingClassifier, self).predict(X)
