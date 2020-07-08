import numpy as np

from wellcomeml.ml import BertClassifier

X = ["Hot and cold", "Hot", "Cold"]
Y = np.array([[1, 1], [1, 0], [0, 1]])

bert = BertClassifier()
bert.fit(X, Y)
print(bert.score(X, Y))
