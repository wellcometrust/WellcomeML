"""Doc2Vec sklearn wrapper"""
import json

from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

class Doc2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window_size=2, n_jobs=1):
        self.vector_size = vector_size
        self.window_size = window_size
        self.n_jobs = n_jobs

    def _yield_tagged_documents(self, X):
        for i, x in enumerate(X):
            yield TaggedDocument(x, [i])

    def fit(self, X):
        # TODO: Implement streaming option
        tagged_documents = list(self._yield_tagged_documents(X))
        self.model = Doc2Vec(tagged_documents, vector_size, window, workers=self.n_jobs)

    def transform(self, X):
        return np.array([self.model.infer_vector(x) for x in X])

    def save():
        pass

    def load():
        pass

if __name__ == '__main__':
    DATA = "data.jsonl"

    def yield_texts(data):
        with open(data) as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                yield TaggedDocument(item["text"], [i])

    texts = yield_texts(DATA)

# Option 1 - Stream
#model = Doc2Vec(vector_size=5, window=2, min_count=1, workers=2)
#model.build_vocab(texts)
#model.train(texts, total_examples=model.corpus_count, epochs=20)

# Option 2 - In memory
    model = Doc2Vec(list(texts), vector_size=5, window=2, min_count=1, workers=2)

    vectors = [model.infer_vector(doc) for doc in ["This is malaria".split()]*5]
    print(vectors)

    print(type(vectors[0]))
    print(np.array(vectors).shape)
