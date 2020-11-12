""" Whole test set and batchwise f1 metrics for use with keras

Adapted from: https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
"""
import tensorflow as tf
import tensorflow.keras.backend as K


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
    return 1 - f1_metric(y_true, y_pred)
