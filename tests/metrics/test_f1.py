#!/usr/bin/env python3
# coding: utf-8

import os
import tempfile

import numpy as np

import pytest
import tensorflow as tf
from wellcomeml.metrics import f1_loss, f1_metric


@pytest.fixture(scope="module")
def tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp("test_f1")


@pytest.fixture(scope="module")
def data():
    X_train = np.random.random((100, 10))
    y_train = np.random.random(100).astype(int)

    X_test = np.random.random((100, 10))
    y_test = np.random.random(100).astype(int)

    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}


@pytest.fixture(scope="module")
def model():
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(1, "sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def test_f1_metric_all_true():

    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 0]

    f1 = f1_metric(y_true, y_pred)

    assert isinstance(f1, tf.Tensor)
    assert f1 == 1.0


def test_f1_metric_all_false():

    y_true = [0, 1, 1, 0]
    y_pred = [0, 0, 0, 0]

    f1 = f1_metric(y_true, y_pred)

    assert isinstance(f1, tf.Tensor)
    assert f1 == 0.0


def test_f1_metric_poor_recall():

    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    f1 = f1_metric(y_true, y_pred)

    assert isinstance(f1, tf.Tensor)
    assert f1 == 0.66666657


def test_f1_metric_poor_precision():

    y_true = [0, 1, 1, 0]
    y_pred = [1, 0, 0, 0]

    f1 = f1_metric(y_true, y_pred)

    assert isinstance(f1, tf.Tensor)
    assert f1 == 0.0


def test_f1_metric(data, model):
    """ Test whether the f1_metrics are output"""

    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=[f1_metric],
    )

    history = model.fit(
        data["X_train"],
        data["y_train"],
        epochs=5,
        validation_data=(data["X_test"], data["y_test"]),
        batch_size=1024,
        verbose=0,
    )

    assert set(history.history.keys()) == set(
        ["loss", "f1_metric", "val_loss", "val_f1_metric"]
    )


def test_f1_loss(data, model):
    """ Test to see if it runs, don't test the loss itself """

    model.compile(
        loss=f1_loss, optimizer="adam", metrics=["accuracy"],
    )

    history = model.fit(
        data["X_train"],
        data["y_train"],
        epochs=5,
        validation_data=(data["X_test"], data["y_test"]),
        batch_size=1024,
        verbose=0,
    )


