from wellcomeml.ml import CNNClassifier, KerasVectorizer
from sklearn.pipeline import Pipeline

import numpy as np

X = ["One", "three", "one", "two", "four"]
Y = np.array([1,0,1,0,0])

cnn_pipeline = Pipeline([
	('vec', KerasVectorizer()),
	('clf', CNNClassifier())
])
cnn_pipeline.fit(X, Y)
print(cnn_pipeline.score(X, Y))

X = ["One, three","one","two, three"]
Y = np.array([[1,0,1],[1,0,0],[0,1,1]])

cnn_pipeline = Pipeline([
	('vec', KerasVectorizer()),
	('clf', CNNClassifier(multilabel=True))
])
cnn_pipeline.fit(X, Y)
print(cnn_pipeline.score(X, Y))