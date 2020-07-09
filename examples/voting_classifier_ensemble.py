from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from wellcomeml.ml.voting_classifier import WellcomeVotingClassifier

X = [
    "One two",
    "One",
    "Three and two",
    "Three"
]
Y = [
    [1, 1, 0],
    [1, 0, 0],
    [0, 1, 1],
    [0, 0, 1]
]

vec = CountVectorizer()
vec.fit(X)

X_vec = vec.transform(X)

sgd = OneVsRestClassifier(SGDClassifier(loss="log"))
nb = OneVsRestClassifier(MultinomialNB())

sgd.fit(X_vec, Y)
nb.fit(X_vec, Y)

voting_classifier = WellcomeVotingClassifier(
    estimators=[sgd, nb], voting="soft", multilabel=True
)

Y_pred = voting_classifier.predict(X_vec)
print(Y_pred)

Y = [1, 0, 1, 0]

sgd = SGDClassifier(loss="log")
nb = MultinomialNB()

sgd.fit(X_vec, Y)
nb.fit(X_vec, Y)

voting_classifier = WellcomeVotingClassifier(
    estimators=[sgd, nb], voting="soft"
)

Y_pred = voting_classifier.predict(X_vec)
print(Y_pred)
