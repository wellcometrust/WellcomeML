"""Doc2Vec sklearn wrapper"""
from pathlib import Path
import multiprocessing
import statistics
import logging

from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

logging.getLogger("gensim").setLevel(logging.WARNING)


class Doc2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        vector_size=100,
        window_size=5,
        n_jobs=1,
        min_count=2,
        negative=5,
        sample=1e-5,
        epochs=20,
        learning_rate=0.025,
        model="dm",
        pretrained=None,
    ):
        """
        Args:
            vector_size: size of vector to represent text
            window_size: words left and right of context words used to create representation
            min_count: filter words that appear less than min_count. default: 2
            negative: number of negative words to be used for training.
                      if zero hierarchical softmax is used. default: 5
            sample: threshold for downsampling high frequency words. default: 1e-5
            learning_rate: learning rate used by SGD. default: 0.025
            model: underlying model architecture, one of dm or dbow. default: dm
            epochs: number of passes over training data. default: 20
            n_jobs: number of cores to use (-1 for all). default: 1
            pretrained: path to directory containing saved pretrained doc2vec artifacts
        """
        self.vector_size = vector_size
        self.window_size = window_size
        self.epochs = epochs
        self.min_count = min_count
        self.negative = negative
        self.sample = sample
        self.n_jobs = n_jobs
        self.learning_rate = learning_rate
        self.model = model
        self.pretrained = pretrained

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
        # If pretrained, just load, no need to fit
        if self.pretrained:
            self.load(self.pretrained)
            return self

        if self.n_jobs == -1:
            workers = multiprocessing.cpu_count()
        else:
            workers = self.n_jobs
        # TODO: Debug streaming implementation below
        #    atm it gives different result than non streaming

        # tagged_documents = self._yield_tagged_documents(X)

        # self.model = Doc2Vec(
        #    vector_size=self.vector_size, window_size=self.window_size,
        #    workers=workers, min_count=self.min_count, epochs=self.epochs
        # )
        # self.model.build_vocab(tagged_documents)
        # self.model.train(tagged_documents, total_examples=self.model.corpus_count,
        #                  epochs=self.model.epochs)
        tagged_documents = list(self._yield_tagged_documents(X))
        self.model = Doc2Vec(
            tagged_documents,
            vector_size=self.vector_size,
            window=self.window_size,
            workers=workers,
            min_count=self.min_count,
            epochs=self.epochs,
            negative=self.negative,
            sample=self.sample,
            alpha=self.learning_rate,
            dm=1 if self.model == "dm" else 0,
            hs=1 if self.negative == 0 else 0,
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
            sims = self.model.docvecs.most_similar(
                [inferred_vector], topn=len(self.model.docvecs)
            )
            rank = [docid for docid, sim in sims].index(doc_id)
            correct.append(int(rank == 0))

        return statistics.mean(correct)

    def _get_model_path(self, model_dir):
        return "{}/doc2vec".format(model_dir)

    def save(self, model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        model_path = self._get_model_path(model_dir)
        self.model.save(model_path)

    def load(self, model_dir):
        model_path = self._get_model_path(model_dir)
        self.model = Doc2Vec.load(model_path)
