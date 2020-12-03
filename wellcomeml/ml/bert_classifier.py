#!/usr/bin/env python
# coding: utf8
"""
Train a convolutional neural network for multilabel classification
of grants
Adapted from https://github.com/explosion/spaCy/blob/master/examples/training/train_textcat.py
"""
from spacy.training import Example
from spacy.util import minibatch, compounding
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import numpy as np
import spacy
import torch

import random
import time

from wellcomeml.logger import logger


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
        self.nlp = spacy.blank("en")
        if self.pretrained == "bert":
            transformer_name = "bert-base-uncased"
        elif self.pretrained == "scibert":
            transformer_name = "allenai/scibert_scivocab_uncased"
        else:
            logger.info(f"{self.pretrained} is not among bert, scibert")
            raise
        self.nlp.add_pipe(
            "transformer",
            config={
                "model": {
                    "@architectures": "spacy-transformers.TransformerModel.v1",
                    "name": transformer_name,
                    "tokenizer_config": {"use_fast": "true"},
                }
            }
        )
        # TODO: Add a parameter for exclusive classes, non multilabel scenario
        self.textcat = self.nlp.add_pipe(
            "textcat",
            config={
                "model": {
                    "@architectures": "spacy.TextCatCNN.v1",
                    "exclusive_classes": False,
                    "tok2vec": {
                        "@architectures": "spacy-transformers.TransformerListener.v1",
                        "grad_factor": 1.0,
                        "pooling": {"@layers": "reduce_mean.v1"}
                    }
                }
            }
        )

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

    def _data_to_examples(self, data):
        """convert list of text, annotation to list of examples"""
        examples = []
        for training_example in data:
            text, annotation = training_example
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annotation)
            examples.append(example)
        return examples

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
        examples = self._data_to_examples(train_data)

        other_pipes = [
            pipe for pipe in self.nlp.pipe_names
            if pipe not in ["textcat", "transformer"]
        ]
        with self.nlp.select_pipes(disable=other_pipes):  # only train textcat and transformer
            optimizer = self.nlp.initialize(lambda: examples)
            optimizer.learn_rate = self.learning_rate
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
                random.shuffle(examples)
                batches = minibatch(examples, size=batch_sizes)
                for batch in batches:
                    # texts, annotations = zip(*batch)
                    next_dropout = self.dropout  # next(dropout)
                    self.nlp.update(
                        examples,
                        sgd=optimizer,
                        drop=next_dropout,
                        losses=losses,
                    )
                    nb_examples += len(batch)
                # FIX
                # with self.textcat.model.use_params(optimizer.averages):
                Y_test_pred = self.predict(X_test)
                p, r, f1, _ = precision_recall_fscore_support(
                        Y_test, Y_test_pred, average="micro"
                    )
                loss = losses["textcat"]
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
        examples = self._data_to_examples(zip(texts, annotations))

        other_pipes = [
            pipe for pipe in self.nlp.pipe_names
            if pipe not in ["transformer", "textcat"]
        ]
        with self.nlp.select_pipes(disable=other_pipes):  # only train textcat and transformer
            if not hasattr(self, "optimizer"):
                self.optimizer = self.nlp.initialize(lambda: examples)
                self.optimizer.learn_rate = self.learning_rate

            losses = {}
            self.nlp.update(
                examples, sgd=self.optimizer, drop=self.dropout, losses=losses
            )
            self.losses.append(losses["textcat"])
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
