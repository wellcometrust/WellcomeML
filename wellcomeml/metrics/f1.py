""" Whole test set and batchwise f1 metrics for use with keras

Adapted from: https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
"""
import numpy as np

import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.callbacks import Callback


def f1_metric(y_true, y_pred):
    """Calculate batchwise macro f1

    >>> model.compile(
    >>>     loss="binary_crossentropy",
    >>>     optimizer="adam",
    >>>     metrics=["accuracy", f1_metric]
    >>> )
    """
    y_true = K.cast(y_true, "float")
    y_pred = K.cast(y_pred, "float")
    tp = K.sum(K.cast(y_true * y_pred, "float"), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, "float"), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), "float"), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)


def f1_loss(y_true, y_pred):
    """Generate batchwise macro f1 loss

    >>> model.compile(
    >>>     loss=f1_loss,
    >>>     optimizer="adam",
    >>>     metrics=["accuracy"]
    >>> )
    """
    y_true = K.cast(y_true, "float")
    y_pred = K.cast(y_pred, "float")
    tp = K.sum(K.cast(y_true * y_pred, "float"), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, "float"), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), "float"), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    return 1 - K.mean(f1)


class GlobalF1(Callback):
    """Calculate global F1, precision, and recall, against a complete test set
    not just batch level metrics as with f1_metric.

    >>> global_f1 = GlobalF1(validation=(X_test, y_test, history.csv))
    >>>
    >>> history = model.fit(
    >>>     X_train,
    >>>     y_train,
    >>>     epochs=100,
    >>>     validation_data=(X_test, y_test),
    >>>     batch_size=1024,
    >>>     verbose=2,
    >>>     callbacks=[early_stopping, glbal_f1],
    >>> )
    """

    def __init__(self, validation, history_path: str = None):
        super(GlobalF1, self).__init__()
        self.validation = validation
        self.history_path = history_path

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_targ = self.validation[1]
        val_predict = (np.asarray(self.model.predict(self.validation[0]))).round()

        val_f1 = f1_score(val_targ, val_predict)
        val_recall = recall_score(val_targ, val_predict)
        val_precision = precision_score(val_targ, val_predict)

        self.val_f1s.append(round(val_f1, 6))
        self.val_recalls.append(round(val_recall, 6))
        self.val_precisions.append(round(val_precision, 6))

        print(
            "Global metrics "
            f"— val_f1: {round(val_f1,4)} "
            f"— val_precision: {round(val_precision, 4)} "
            f"— val_recall: {round(val_recall,4)}"
        )

    def on_train_end(self, logs={}):
        """Write metrics to csv file"""

        if self.history_path:
            history = {}
            history["f1"] = self.val_f1s
            history["precision"] = self.val_precisions
            history["recall"] = self.val_recalls
            history_df = pd.DataFrame(history)
            history_df.to_csv(self.history_path, index_label="epoch")
