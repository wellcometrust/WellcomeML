from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs):
        X_val = self.validation_data[0]
        Y_val = self.validation_data[1]

        Y_pred = self.model.predict(X_val) > 0.5
        f1 = round(f1_score(Y_val, Y_pred, average='micro'), 4)
        p = round(precision_score(Y_val, Y_pred, average='micro'), 4)
        r = round(recall_score(Y_val, Y_pred, average='micro'), 4)
        print(f" - val metrics: P {p:.4f} R {r:.4f} F {f1:.4f}")
        return


class CategoricalMetrics(tf.keras.metrics.Metric):
    def __init__(self, metric='precision', from_logits=True,
                 threshold=0.5,
                 binary='true', pos_label=1, **kwargs):
        """
        Categorical metrics

        Args:
            metric: 'precision', 'recall' or 'f1'
            from_logits: Similar to keras 'from_logits'. If True, calculates
                        a softmax activation on input
            threshold: a threshold to calculate predictions
            binary: whether input is binary or not (in which case does average)
            pos_label: the positive label
        """
        self.metric = metric
        name = f"categorical_{metric}"
        super(CategoricalMetrics, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.from_logits = from_logits
        self.binary = binary
        self.pos_label = pos_label
        self.value = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = tf.keras.activations.softmax(y_pred)

        greater_than_threshold = tf.cast(y_pred[:, 1:] > self.threshold, 'bool')
        positive = tf.cast(y_true, 'int32') == self.pos_label

        # Epsilon added to denominators to avoid division by zero
        eps = tf.keras.backend.epsilon()

        tp = tf.reduce_sum(
            tf.cast(
                tf.logical_and(greater_than_threshold, positive),
                'float32'
            )
        )

        if self.metric in ['precision', 'f1_score']:
            tp_plus_fp = tf.reduce_sum(
                tf.cast(
                    greater_than_threshold,
                    'float32'
                )
            )
            precision = tp/(tp_plus_fp+eps)

        if self.metric in ['recall', 'f1_score']:
            tp_plus_fn = tf.reduce_sum(
                tf.cast(
                    positive,
                    'float32'
                )
            )
            recall = tp/(tp_plus_fn+eps)

        if self.metric == 'precision':
            self.value.assign(precision)
        elif self.metric == 'recall':
            self.value.assign(recall)
        elif self.metric == 'f1_score':
            self.value.assign(
                2*precision*recall/(precision+recall+eps)
            )

    def result(self):
        return self.value

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.value.assign(0.)
