import csv
from collections import defaultdict

from sklearn.metrics import f1_score, precision_score, recall_score

from wellcomeml.utils import throw_extra_import_message

try:
    import tensorflow as tf
except ImportError as e:
    throw_extra_import_message(error=e, required_module='tensorflow', extra='tensorflow')


class Metrics(tf.keras.callbacks.Callback):
    """Calculate global F1, precision, and recall, against a complete test set
    not just batch level metrics as with f1_metric.

    >>> metrics = Metrics(validation_data=(X_test, y_test), history.csv)
    >>>
    >>> history = model.fit(
    >>>     X_train,
    >>>     y_train,
    >>>     epochs=100,
    >>>     validation_data=(X_test, y_test),
    >>>     batch_size=1024,
    >>>     verbose=2,
    >>>     callbacks=[early_stopping, metrics],
    >>> )
    """

    def __init__(
        self,
        validation_data,
        history_path: str = None,
        append: bool = False,
        average: str = "micro",
    ):
        self.validation_data = validation_data
        self.history_path = history_path
        self.append = append
        self.average = average

    def on_train_begin(self, logs={}):
        self.f1s = []
        self.recalls = []
        self.precisions = []

    def on_epoch_end(self, epoch, logs):
        X_val = self.validation_data[0]
        Y_val = self.validation_data[1]

        Y_pred = self.model.predict(X_val) > 0.5
        f1 = round(f1_score(Y_val, Y_pred, average=self.average), 4)
        p = round(precision_score(Y_val, Y_pred, average=self.average), 4)
        r = round(recall_score(Y_val, Y_pred, average=self.average), 4)

        self.f1s.append(f1)
        self.recalls.append(r)
        self.precisions.append(p)

        print(f" - val metrics: P {p:.4f} R {r:.4f} F {f1:.4f}")

        return

    def on_train_end(self, logs={}):
        """Write metrics to csv file"""

        if self.history_path:
            mode = "w"
            header = True

            if self.append:
                mode = "a"
                header = False

            with open(self.history_path, mode=mode) as fb:
                history_writer = csv.writer(fb, delimiter=",")
                if header:
                    history_writer.writerow(["epoch", "precision", "recall", "f1"])

                for i, row in enumerate(zip(self.precisions, self.recalls, self.f1s)):
                    history_writer.writerow([i] + list(row))


class CategoricalMetrics(tf.keras.metrics.Metric):
    def __init__(
        self,
        metric="precision",
        from_logits=True,
        threshold=0.5,
        binary="true",
        pos=1,
        **kwargs,
    ):
        """
        Categorical metrics

        Args:
            metric: 'precision', 'recall' or 'f1'
            from_logits: Similar to keras 'from_logits'. If True, calculates
                a softmax activation on input
            threshold: a threshold to calculate predictions
            binary: whether input is binary or not (in which case does average)
            pos: the positive label (between [0, n_classes])
            **kwargs:
        """
        self.metric = metric
        name = f"categorical_{metric}"
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.from_logits = from_logits
        self.binary = binary
        self.pos = pos
        self.value = self.add_weight(name="tp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = tf.keras.activations.softmax(y_pred)

        greater_than_threshold = tf.cast(y_pred[:, self.pos:] > self.threshold, "bool")
        positive = tf.cast(y_true, "int32") == self.pos

        # Epsilon added to denominators to avoid division by zero
        eps = tf.keras.backend.epsilon()

        tp = tf.reduce_sum(
            tf.cast(tf.logical_and(greater_than_threshold, positive), "float32")
        )

        if self.metric in ["precision", "f1_score"]:
            tp_plus_fp = tf.reduce_sum(tf.cast(greater_than_threshold, "float32"))
            precision = tp / (tp_plus_fp + eps)

        if self.metric in ["recall", "f1_score"]:
            tp_plus_fn = tf.reduce_sum(tf.cast(positive, "float32"))
            recall = tp / (tp_plus_fn + eps)

        if self.metric == "precision":
            self.value.assign(precision)
        elif self.metric == "recall":
            self.value.assign(recall)
        elif self.metric == "f1_score":
            self.value.assign(2 * precision * recall / (precision + recall + eps))

    def result(self):
        return self.value

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.value.assign(0.0)


class MetricMiniBatchHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.metric_history_mini_batch = defaultdict(list)

    def on_batch_end(self, batch, logs={}):
        for metric in self.params["metrics"]:
            if logs.get(metric):
                self.metric_history_mini_batch[metric].append(logs.get(metric))
                self.model.history.history[
                    f"mini_batch_{metric}"
                ] = self.metric_history_mini_batch[metric]
