from wellcomeml.ml.bilstm import BiLSTMClassifier
from wellcomeml.ml.keras_vectorizer import  KerasVectorizer
from sklearn.pipeline import Pipeline

import numpy as np

X = ["One", "three", "one", "two", "four"]
Y = np.array([1, 0, 1, 0, 0])

bilstm_pipeline = Pipeline([("vec", KerasVectorizer()), ("clf", BiLSTMClassifier())])
bilstm_pipeline.fit(X, Y)
print(bilstm_pipeline.score(X, Y))

X = ["One, three", "one", "two, three"]
Y = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 1]])

bilstm_pipeline = Pipeline(
    [("vec", KerasVectorizer()), ("clf", BiLSTMClassifier(multilabel=True))]
)
bilstm_pipeline.fit(X, Y)
print(bilstm_pipeline.score(X, Y))
