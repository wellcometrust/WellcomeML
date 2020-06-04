from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


from wellcomeml.ml.sent2vec_vectorizer import Sent2VecVectorizer

X = [
    "Malaria is a disease spread by mosquitos",
    "HIV is a virus that causes a disease named AIDS",
    "Trump is the president of USA"
]

sent2vec = Sent2VecVectorizer()
X_transformed = sent2vec.fit_transform(X)
print(cosine_similarity(X_transformed))

y = [1,1,0]

model = Pipeline([
    ('sent2vec', Sent2VecVectorizer()),
    ('sgd', SGDClassifier())
])
model.fit(X, y)
