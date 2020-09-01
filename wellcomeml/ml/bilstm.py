from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
import tensorflow as tf

from wellcomeml.ml.attention import HierarchicalAttention

TENSORBOARD_LOG_DIR = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
METRIC_DICT = {
    'precision': tf.keras.metrics.Precision(name='precision'),
    'recall': tf.keras.metrics.Recall(name='recall'),
    'auc': tf.keras.metrics.AUC(name='auc')
}
CALLBACK_DICT = {
    'tensorboard': tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOG_DIR)
}


class BiLSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        learning_rate=0.01,
        batch_size=32,
        nb_epochs=5,
        dropout=0.1,
        nb_layers=2,
        hidden_size=100,
        dense_size=20,
        l2=1e-6,
        attention_heads='same',
        multilabel=False,
        attention=False,
        metrics=["precision", "recall"],
        callbacks=["tensorboard"],
        feature_approach="max"
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.dropout = dropout
        self.nb_layers = nb_layers
        self.hidden_size = hidden_size
        self.l2 = l2
        self.attention_heads = attention_heads
        self.dense_size = dense_size
        self.multilabel = multilabel
        self.attention = attention
        self.metrics = metrics
        self.callbacks = callbacks
        self.feature_approach = feature_approach

    def _build_model(self, sequence_length, vocab_size, nb_outputs,
                     embedding_matrix=None, metrics=["precision", "recall"]):
        output_activation = (
            "sigmoid" if nb_outputs == 1 or self.multilabel else "softmax"
        )
        embedding_size = embedding_matrix.shape[1] if embedding_matrix else self.hidden_size

        def residual_bilstm(x1, l2):
            x2 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    int(x1.shape[-1] / 2), return_sequences=True, kernel_regularizer=l2
                )
            )(x1)
            return tf.keras.layers.add([x1, x2])

        def residual_attention(x1):
            x2 = HierarchicalAttention(self.attention_heads)(x1)
            x2 = tf.keras.layers.Dropout(self.dropout)(x2)
            x2 = tf.keras.layers.LayerNormalization()(x2)
            if self.attention_heads == 'same':
                return tf.keras.layers.add([x1, x2])
            else:
                return x2

        l2 = tf.keras.regularizers.l2(self.l2)
        embeddings_initializer = (
            tf.keras.initializers.Constant(embedding_matrix)
            if embedding_matrix
            else "uniform"
        )
        inp = tf.keras.layers.Input(shape=(sequence_length,))
        x = tf.keras.layers.Embedding(
            vocab_size,
            embedding_size,
            input_length=sequence_length,
            embeddings_initializer=embeddings_initializer,
        )(inp)
        x = tf.keras.layers.Dropout(
            self.dropout,
            noise_shape=(None, sequence_length, 1))(x)
        for _ in range(self.nb_layers):
            x = residual_bilstm(x, l2)
        if self.attention:
            x = residual_attention(x)
        if self.feature_approach == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        elif self.feature_approach == 'sum':
            x = tf.keras.backend.sum(x, axis=1)
        elif self.feature_approach == 'concat':
            x = tf.concat(x, axis=1)
        else:
            raise NotImplementedError
        x = tf.keras.layers.Dense(self.dense_size, kernel_regularizer=l2)(x)
        out = tf.keras.layers.Dense(
            nb_outputs, activation=output_activation, kernel_regularizer=l2
        )(x)
        model = tf.keras.Model(inp, out)

        optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate, clipnorm=1.0)
        metrics = [
            METRIC_DICT[m] if m in METRIC_DICT else m
            for m in metrics
        ]
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)
        return model

    def fit(self, X, Y, embedding_matrix=None, *_):
        sequence_length = X.shape[1]
        vocab_size = X.max() + 1

        nb_outputs = Y.max() if not self.multilabel else Y.shape[1]

        self.model = self._build_model(sequence_length, vocab_size, nb_outputs,
                                       embedding_matrix, self.metrics)

        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=0.1, shuffle=True
        )
        callbacks = [
            CALLBACK_DICT[c] if c in CALLBACK_DICT else c
            for c in self.callbacks
        ]
        self.model.fit(
            X_train,
            Y_train,
            epochs=self.nb_epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, Y_val),
            callbacks=callbacks,
        )
        return self

    def predict(self, X, *_):
        return self.model(X).numpy() > 0.5

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return f1_score(Y, Y_pred, average="micro")
