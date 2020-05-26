"""Doc2Vec sklearn wrapper"""
from collections import Counter
import multiprocessing
import statistics
import json

from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

class Doc2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window_size=5, n_jobs=1,
                 min_count=2, epochs=20):
        """
        Args:
            vector_size: size of vector to represent text
            window_size: words left and right of context words used to create representation
            min_count: filter words that appear less than min_count. default: 2
            epochs: number of passes over training data. default: 20
            n_jobs: number of cores to use (-1 for all). default: 1
        """
        self.vector_size = vector_size
        self.window_size = window_size
        self.epochs = epochs
        self.min_count = min_count
        self.n_jobs = n_jobs

    def _tokenize(self, x):
        return x.lower().split()

    def _yield_tagged_documents(self, X):
        for i, x in enumerate(X):
            yield TaggedDocument(self._tokenize(x), [i])

    def fit(self, X, *_):
        """
        Args:
            X: list of texts (strings)
        """
        if self.n_jobs == -1:
            workers = multiprocessing.cpu_count()
        else:
            workers = self.n_jobs
        # TODO: Debug streaming implementation below
        #    atm it gives different result than non streaming

        #tagged_documents = self._yield_tagged_documents(X)
            
        #self.model = Doc2Vec(
        #    vector_size=self.vector_size, window_size=self.window_size,
        #    workers=workers, min_count=self.min_count, epochs=self.epochs
        #)
        #self.model.build_vocab(tagged_documents)
        #self.model.train(tagged_documents, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        tagged_documents = list(self._yield_tagged_documents(X))
        self.model = Doc2Vec(
            tagged_documents, vector_size=self.vector_size,
            window_size=self.window_size, workers=workers,
            min_count=self.min_count, epochs=self.epochs
        )
        return self

    def transform(self, X):
        """
        Args:
            X: list of texts (strings)
        Returns:
            docvectors: matrix of size (nb_docs, vector_size)
        """
        return np.array([self.model.infer_vector(self._tokenize(x)) for x in X])

    def score(self, X):
        """
        Args:
            X: list of texts (strings). Needs to be the same used for fit.
        Returns:
            score: percentage of documents that are most similar with themselves
        """
        correct = []

        docvecs = self.transform(X)
        for doc_id, inferred_vector in enumerate(docvecs):
            sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
            rank = [docid for docid, sim in sims].index(doc_id)
            correct.append(int(rank==0))
        
        return statistics.mean(correct)

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

    doc2vec = Doc2VecVectorizer(epochs=40)
    doc2vec.fit(texts)

    print(doc2vec.score(texts))

#    doc2vec.save("delete_doc2vec_model")
