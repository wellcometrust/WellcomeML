from wellcomeml.ml import SpacyClassifier

import numpy as np

X = ["One, three","one","two, three"]
Y = [[1,0,1],[1,0,0],[0,1,1]]

spacy_classifier = SpacyClassifier()
spacy_classifier.fit(X, Y)
print(spacy_classifier.score(X, Y))
