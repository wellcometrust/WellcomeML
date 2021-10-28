from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from wellcomeml.ml.bert_vectorizer import BertVectorizer


X = [
    "Elizabeth is the queen of England",
    "Felipe is the king of Spain",
    "I like to travel",
]
y = [1, 1, 0]

vectorizer = BertVectorizer()
X_transformed = vectorizer.fit_transform(X)
print(cosine_similarity(X_transformed))

pipeline = Pipeline([("bert", BertVectorizer()), ("svm", SVC(kernel="linear"))])
pipeline.fit(X, y)
print(pipeline.score(X, y))
