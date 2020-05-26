from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


from wellcomeml.ml import Doc2VecVectorizer

X = [
    "Malaria is a disease spread by mosquitos",
    "HIV is a virus that causes a disease named AIDS",
    "Trump is the president of USA"
]

doc2vec = Doc2VecVectorizer(min_count=1, vector_size=8)
X_transformed = doc2vec.fit_transform(X)
print(cosine_similarity(X_transformed))

y = [1,1,0]

model = Pipeline([
    ('doc2vec', Doc2VecVectorizer(min_count=1, vector_size=8)),
    ('sgd', SGDClassifier())
])
model.fit(X, y)
print(model.score(X, y))
