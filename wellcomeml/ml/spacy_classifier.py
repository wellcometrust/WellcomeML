#!/usr/bin/env python
# coding: utf8
"""
Train a convolutional neural network for multilabel classification
of grants
Adapted from https://github.com/explosion/spaCy/blob/master/examples/training/train_textcat.py
"""
from spacy.util import minibatch, compounding
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from scipy.sparse import csr_matrix
import numpy as np
import spacy
import torch

import random
import time

from wellcomeml.logger import logger

is_using_gpu = spacy.prefer_gpu()
if is_using_gpu:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")


class SpacyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        threshold=0.5,
        n_iterations=5,
        batch_size=8,
        learning_rate=0.001,
        dropout=0.1,
        shuffle=True,
        architecture="simple_cnn",
        multilabel=True,
        pre_trained_vectors_path=None,
    ):
        """
        Args:
            threshold: the threshold above of which a label should be assigned.
                Should take values from 0 to 1. default is 0.5.
            batch_size: the number of examples that will be given to the optimizer
                in each update of the parameters. default is 8.
            dropout: the dropout added to the layers. default is 0.1.
            learning_rate: the learning rate of the optimizer. default is 0.001.
            n_iterations: number of iterations on the entire dataset (epochs)
                default is 5
            shuffle: whether the example are shuffled before an iterations.
                default is True
            architecture: architecture that Spacy uses. can be one of "bow", "simple_cnn"
                and "ensemble". default is "simple_cnn"
            multilabel: whether the problem is multilabel and thus Y is a matrix. default is True
            pre_trained_vectors_path: path to pretrained vectors produced by spacy pretrain.
                default is None
        """
        self.threshold = threshold
        self.batch_size = batch_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.shuffle = shuffle
        self.architecture = architecture
        self.pre_trained_vectors_path = pre_trained_vectors_path
        self.multilabel = multilabel

    def _init_nlp(self):
        self.nlp = spacy.blank("en")

        self.textcat = self.nlp.create_pipe(
            "textcat",
            config={
                "architecture": self.architecture,
                "exclusive_classes": not self.multilabel,
            },
        )
        self.nlp.add_pipe(self.textcat, last=True)

        for label in self.unique_labels:
            self.textcat.add_label(label)

    def load(self, model_dir):
        self.nlp = spacy.load(model_dir)
        # another hack to get the labels from textcat since
        # spacy does not serialise every attribute
        for pipe_name, pipe in self.nlp.pipeline:
            # to cover case of textcat and trf_textcat
            if "textcat" in pipe_name:
                self.unique_labels = pipe.labels

    def save(self, output_dir):
        self.nlp.to_disk(output_dir)

    def _label_binarizer_inverse_transform(self, Y_train):
        "Transforms Y matrix to labels which are the non zero indices"
        data = []
        for row in Y_train:
            if type(row) is csr_matrix:
                row = row.todense()
                row = np.squeeze(np.asarray(row))
            row_data = [str(i) for i, item in enumerate(row) if item]
            data.append(row_data)
        # Do we need to convert to numpy?
        return np.array(data)

    def fit(self, X, Y):
        """
        Args:
            X: 1d list of documents
            Y: 2d numpy array (nb_examples, nb_labels)
        TODO: Generalise to y being 1d
        """
        if type(Y) in [list, tuple]:
            Y = np.array(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)
        # Free memory
        del X
        del Y

        if self.multilabel:
            nb_labels = Y_train.shape[1]
            self.unique_labels = [str(i) for i in range(nb_labels)]
        else:
            self.unique_labels = np.unique(Y_train)
        self._init_nlp()
        n_iter = self.n_iterations

        def yield_train_data(X_train, Y_train):
            for x, y in zip(X_train, Y_train):
                if self.multilabel:
                    tags = self._label_binarizer_inverse_transform([y])[0]
                    cats = {label: label in tags for label in self.unique_labels}
                else:
                    cats = {label: label == y for label in self.unique_labels}
                yield (x, {"cats": cats})

        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "textcat"]
        with self.nlp.disable_pipes(*other_pipes):  # only train textcat
            optimizer = self.nlp.begin_training()
            optimizer.alpha = self.learning_rate
            # optimizer.L2 = 1e-4

            if self.pre_trained_vectors_path:
                with open(self.pre_trained_vectors_path, "rb") as f:
                    self.textcat.model.tok2vec.from_bytes(f.read())

            logger.info("Training the model...")
            logger.info(
                "{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}".format(
                    "ITER", "LOSS", "P", "R", "F", "SPEED"
                )
            )
            batch_sizes = compounding(4.0, self.batch_size, 1.001)
            # dropout = decaying(0.6, 0.2, 1e-4)
            for i in range(n_iter):
                start_time = time.time()
                nb_examples = 0
                losses = {}

                def shuffle(X_train, Y_train):
                    # this is not memory friendly but there should be
                    # memory from deleting X, Y
                    d = list(zip(X_train, Y_train))
                    random.shuffle(d)
                    X_train, Y_train = zip(*d)
                    return X_train, Y_train

                if self.shuffle:
                    X_train, Y_train = shuffle(X_train, Y_train)

                train_data = yield_train_data(X_train, Y_train)
                batches = minibatch(train_data, size=batch_sizes)
                for batch in batches:
                    texts, annotations = zip(*batch)
                    next_dropout = self.dropout  # next(dropout)
                    self.nlp.update(
                        texts,
                        annotations,
                        sgd=optimizer,
                        drop=next_dropout,
                        losses=losses,
                    )
                    nb_examples += len(texts)
                end_time = time.time() - start_time
                examples_per_second = round(nb_examples / end_time, 2)
                with self.textcat.model.use_params(optimizer.averages):
                    Y_test_pred = self.predict(X_test)
                    p, r, f, _ = precision_recall_fscore_support(
                        Y_test, Y_test_pred, average="micro"
                    )
                loss = losses["textcat"]
                logger.info(
                    "{0:2d}\t\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3f}\t{5:.2f} ex/s".format(
                        i, loss, p, r, f, examples_per_second
                    )
                )
        return self

    def partial_fit(self, X, Y, classes=None):
        """
        Args:
            X: 1d list of documents
            Y: 2d numpy array (nb_examples, nb_labels)
        TODO: Generalise to y being 1d
        """
        Y = np.array(Y)

        if not hasattr(self, "unique_labels"):
            # We could also use classes here
            nb_labels = Y.shape[1]
            self.unique_labels = [str(i) for i in range(nb_labels)]
            self._init_nlp()

        texts = X

        if self.multilabel:
            train_tags = self._label_binarizer_inverse_transform(Y)
            train_cats = [
                {label: label in tags for label in self.unique_labels}
                for tags in train_tags
            ]
        else:
            train_cats = []
            for y in Y:
                cats = {label: label == y for label in self.unique_labels}
                train_cats.append(cats)
        annotations = [{"cats": cats} for cats in train_cats]

        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "textcat"]
        with self.nlp.disable_pipes(*other_pipes):  # only train textcat
            if not hasattr(self, "optimizer"):
                self.optimizer = self.nlp.begin_training()
                self.optimizer.alpha = self.learning_rate

            self.nlp.update(texts, annotations, sgd=self.optimizer, drop=self.dropout)

        return self

    def predict(self, X):
        def transform_output(doc):
            cats = doc.cats
            if self.multilabel:
                out = [
                    1 if cats[label] > self.threshold else 0
                    for label in self.unique_labels
                ]
            else:
                out = max(cats.items(), key=lambda x: x[1])[0]
            return out

        docs = self.nlp.pipe(X)
        return np.array([transform_output(doc) for doc in docs])

    def predict_proba(self, X):
        def get_proba(x):
            cats = self.nlp(x).cats
            out = [cats[label] for label in self.unique_labels]
            return out

        return np.array([get_proba(x) for x in X])

    def score(self, X, Y):
        Y = np.array(Y)
        Y_pred = self.predict(X)
        return f1_score(Y_pred, Y, average="micro")
