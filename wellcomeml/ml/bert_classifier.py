#!/usr/bin/env python
# coding: utf8
"""
Train a convolutional neural network for multilabel classification
of grants
Adapted from https://github.com/explosion/spaCy/blob/master/examples/training/train_textcat.py
"""
from spacy_transformers import (
    TransformersLanguage,
    TransformersWordPiecer,
    TransformersTok2Vec,
)
from spacy_transformers.model_registry import (
    register_model,
    get_last_hidden,
    flatten_add_lengths,
)
from spacy.util import minibatch, compounding
from spacy._ml import zero_init, logistic
from thinc.t2v import Pooling, mean_pool
from thinc.v2v import Affine
from thinc.api import chain
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import numpy as np
import spacy
import torch

import random
import time

from wellcomeml.logger import logger


@register_model("sigmoid_last_hidden")
def sigmoid_last_hidden(nr_class, *, exclusive_classes=False, **cfg):
    width = cfg["token_vector_width"]
    return chain(
        get_last_hidden,
        flatten_add_lengths,
        Pooling(mean_pool),
        zero_init(Affine(nr_class, width, drop_factor=0.0)),
        logistic,
    )


is_using_gpu = spacy.prefer_gpu()
if is_using_gpu:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")


class BertClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        threshold=0.5,
        n_iterations=5,
        pretrained="bert",
        batch_size=8,
        learning_rate=1e-5,
        dropout=0.1,
        l2=1e-4,
        validation_split=0.1,
    ):
        self.threshold = threshold
        self.batch_size = batch_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.pretrained = pretrained
        self.l2 = l2
        self.validation_split = validation_split

    def _init_nlp(self):
        if self.pretrained == "bert":
            self.nlp = spacy.load("en_trf_bertbaseuncased_lg")
        elif self.pretrained == "scibert":
            name = "scibert-scivocab-uncased"
            path = "models/scibert_scivocab_uncased"

            nlp = TransformersLanguage(trf_name=name, meta={"lang": "en"})
            nlp.add_pipe(nlp.create_pipe("sentencizer"))
            nlp.add_pipe(TransformersWordPiecer.from_pretrained(nlp.vocab, path))
            nlp.add_pipe(TransformersTok2Vec.from_pretrained(nlp.vocab, path))
            self.nlp = nlp
        else:
            logger.info(f"{self.pretrained} is not among bert, scibert")
            raise
        # TODO: Add a parameter for exclusive classes, non multilabel scenario
        self.textcat = self.nlp.create_pipe(
            "trf_textcat",
            config={"exclusive_classes": False, "architecture": "sigmoid_last_hidden"},
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
            row_data = [str(i) for i, item in enumerate(row) if item]
            data.append(row_data)
        # Do we need to convert to numpy?
        return np.array(data)

    def fit(self, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, random_state=42, test_size=self.validation_split
        )
        self.unique_labels = [str(i) for i in range(Y_train.shape[1])]
        self._init_nlp()
        n_iter = self.n_iterations
        train_texts = X_train
        train_tags = self._label_binarizer_inverse_transform(Y_train)

        train_cats = [
            {label: label in tags for label in self.unique_labels}
            for tags in train_tags
        ]

        train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))

        other_pipes = [pipe for pipe in self.nlp.pipe_names if "trf" not in pipe]
        with self.nlp.disable_pipes(*other_pipes):  # only train textcat
            optimizer = self.nlp.resume_training()
            optimizer.alpha = self.learning_rate
            optimizer.L2 = self.l2
            logger.info("Training the model...")
            logger.info(
                "{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}".format(
                    "ITER", "LOSS", "P", "R", "F", "TF"
                )
            )
            batch_sizes = compounding(4.0, self.batch_size, 1.001)
            # dropout = decaying(0.6, 0.2, 1e-4)
            self.losses = []
            for i in range(n_iter):
                start_epoch = time.time()
                nb_examples = 0
                losses = {}
                # batch up the examples using spaCy's minibatch
                random.shuffle(train_data)
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
                with self.textcat.model.use_params(optimizer.averages):
                    Y_test_pred = self.predict(X_test)
                    p, r, f1, _ = precision_recall_fscore_support(
                        Y_test, Y_test_pred, average="micro"
                    )
                loss = losses["trf_textcat"]
                self.losses.append(loss)

                epoch_seconds = time.time() - start_epoch
                speed = nb_examples / epoch_seconds
                logger.info(
                    "{0:5d}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3f}\t{5:.2f}".format(
                        i, loss, p, r, f1, speed
                    )
                )
        return self

    def partial_fit(self, X, Y, classes=None):
        if not hasattr(self, "unique_labels"):
            self.unique_labels = [str(i) for i in range(Y.shape[1])]
            self._init_nlp()
            self.losses = []

        texts = X

        train_tags = self._label_binarizer_inverse_transform(Y)
        train_cats = [
            {label: label in tags for label in self.unique_labels}
            for tags in train_tags
        ]
        annotations = [{"cats": cats} for cats in train_cats]

        other_pipes = [pipe for pipe in self.nlp.pipe_names if "trf" not in pipe]
        with self.nlp.disable_pipes(*other_pipes):  # only train textcat
            if not hasattr(self, "optimizer"):
                self.optimizer = self.nlp.resume_training()
                self.optimizer.alpha = self.learning_rate

            losses = {}
            self.nlp.update(
                texts, annotations, sgd=self.optimizer, drop=self.dropout, losses=losses
            )
            self.losses.append(losses["trf_textcat"])
        return self

    def predict(self, X):
        def binarize_output(doc):
            cats = doc.cats
            out = [
                1 if cats[label] > self.threshold else 0 for label in self.unique_labels
            ]
            return out
        doc_gen = self.nlp.pipe(X)
        return np.array([binarize_output(doc) for doc in doc_gen])

    def predict_proba(self, X):
        def get_proba(doc):
            cats = doc.cats
            out = [cats[label] for label in self.unique_labels]
            return out
        doc_gen = self.nlp.pipe(X)
        return np.array([get_proba(doc) for doc in doc_gen])

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return f1_score(Y_pred, Y, average="micro")
