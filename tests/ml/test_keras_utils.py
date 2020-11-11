#!/usr/bin/env python3
# coding: utf-8

import os
import tempfile

import numpy as np

import pytest
import tensorflow as tf
from wellcomeml.ml.keras_utils import Metrics


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


def test_metrics_callback(data, model, tmpdir):

    history_path = os.path.join(tmpdir, "test_f1.csv")

    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"],
    )

    metrics = Metrics(
        validation_data=(data["X_test"], data["y_test"]), history_path=history_path
    )

    history = model.fit(
        data["X_train"],
        data["y_train"],
        epochs=5,
        validation_data=(data["X_test"], data["y_test"]),
        batch_size=1024,
        verbose=0,
        callbacks=[metrics],
    )

    assert os.path.exists(history_path)
