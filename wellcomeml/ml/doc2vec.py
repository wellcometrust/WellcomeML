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
        #    model.build_vocab(X)
        #    model.train(X, total_examples=model.corpus_count, epochs=20)
        tagged_documents = list(self._yield_tagged_documents(X))
        self.model = Doc2Vec(
            tagged_documents, vector_size=self.vector_size,
            window_size=self.window_size, workers=self.n_jobs
        )

    def transform(self, X):
        return np.array([self.model.infer_vector(x.split()) for x in X])

    def save(self, model_path):
        self.model.save(model_path)

    def load(self, model_path):
        self.model = Doc2Vec.load(model_path)

if __name__ == '__main__':
    DATA = "data.jsonl"

    def yield_texts(data):
        with open(data) as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                yield item["text"]

    texts = list(yield_texts(DATA))

    doc2vec = Doc2VecVectorizer()
    doc2vec.fit(texts)
    print(doc2vec.transform(texts[:5]))

    doc2vec.save("delete_doc2vec_model")
