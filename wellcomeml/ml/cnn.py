"""
CNN architecture inspired from spacy for NLP tasks

It follows the embed, encode, attend, predict framework

    Embed: learns a new embedding but can receive
    pre trained embeddings as well

    Encode: stacked CNN with context window 3 that maintains the size of input and applies dropout
    and layer norm

    Attend: not yet implemented

    Predict: softmax or sigmoid depending on number of outputs
    and whether task is multilabel
"""
from datetime import datetime
import logging
import math

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix, vstack
import numpy as np

from wellcomeml.ml.attention import HierarchicalAttention
from wellcomeml.utils import throw_extra_import_message

try:
    import tensorflow_addons as tfa
    import tensorflow as tf
except ImportError as e:
    throw_extra_import_message(error=e, required_module='tensorflow', extra='tensorflow')

logger = logging.getLogger(__name__)


class CNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        context_window=3,
        learning_rate=0.001,
        learning_rate_decay=1,
        batch_size=32,
        nb_epochs=5,
        dropout=0.2,
        nb_layers=4,
        hidden_size=100,
        l2=1e-6,
        dense_size=32,
        multilabel=False,
        attention=False,
        attention_heads='same',
        metrics=["precision", "recall", "f1"],
        feature_approach="max",
        early_stopping=False,
        sparse_y=False,
        threshold=0.5,
        validation_split=0.1,
        sequence_length=None,
        vocab_size=None,
        nb_outputs=None,
        tensorboard_log_path="logs",
        verbose=1  # this follows keras.Model.fit verbose for now
    ):
        self.context_window = context_window
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.dropout = dropout
        self.nb_layers = nb_layers
        # note that on current implementation CNN use same hidden size as embedding
        # so if embedding matrix is passed, this is not used. in the future we can decouple
        self.hidden_size = hidden_size
        self.l2 = l2
        self.dense_size = dense_size
        self.multilabel = multilabel
        self.attention = attention
        self.attention_heads = attention_heads
        self.metrics = metrics
        self.feature_approach = feature_approach
        self.early_stopping = early_stopping
        self.sparse_y = sparse_y
        self.threshold = threshold
        self.validation_split = validation_split
        self.sequence_length = sequence_length,
        self.vocab_size = vocab_size,
        self.nb_outputs = nb_outputs,
        self.tensorboard_log_path = tensorboard_log_path,
        self.verbose = verbose

    def _prepare_data(self, X, Y, shuffle=True):
        def yield_data():
            for i in range(X.shape[0]):
                x = X[i]
                y = Y[i, :].todense()  # returns matrix
                y = np.squeeze(np.asarray(y))
                yield x, y

        if self.sparse_y:
            data = tf.data.Dataset.from_generator(yield_data, output_types=(tf.int32, tf.int32))
        else:
            data = tf.data.Dataset.from_tensor_slices((X, Y))

        if shuffle:
            data = data.shuffle(1000)

        data = data.batch(self.batch_size)
        return data

    def _get_distributed_strategy(self):
        if len(tf.config.list_physical_devices('GPU')) > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:  # use default strategy
            strategy = tf.distribute.get_strategy()
        return strategy

    def _init_from_data(self, X, Y):
        logger.info(
            "Initializing sequence_length, vocab_size \
            and nb_outputs from data. This might take a while."
        )
        if isinstance(X, np.ndarray):
            X_max = X.max()
            Y_max = Y.max()
            X_shape = X.shape[1]
            Y_shape = Y.shape[1] if self.multilabel else Y.shape[0]
            steps_per_epoch = math.ceil(X.shape[0] / self.batch_size)
        elif isinstance(X, tf.data.Dataset):
            X = X.batch(self.batch_size)

            # init from iterating over dataset
            X_max = 0
            Y_max = 0
            X_shape = None
            Y_shape = None
            steps_per_epoch = 0
            for X_batch, Y_batch in X:
                batch_X_max = X_batch.numpy().max()
                batch_Y_max = Y_batch.numpy().max()
                if batch_X_max > X_max:
                    X_max = batch_X_max
                if batch_Y_max > Y_max:
                    Y_max = batch_Y_max
                if not Y_shape:
                    Y_shape = Y_batch.shape[1] if self.multilabel else Y_batch.shape[0]
                if not X_shape:
                    X_shape = X_batch.shape[1]
                steps_per_epoch += 1
        else:
            logger.error("CNN currently supports X to one of np.ndarray or tf.data.Dataset")
            raise NotImplementedError

        self.sequence_length = X_shape
        self.vocab_size = X_max + 1
        self.nb_outputs = Y_max if not self.multilabel else Y_shape
        logger.info(
            f"Initialized sequence_length: {self.sequence_length}, \
            vocab_size: {self.vocab_size}, nb_outputs: {self.nb_outputs}"
        )
        return steps_per_epoch

    def _build_model(self, sequence_length, vocab_size, nb_outputs,
                     steps_per_epoch, embedding_matrix=None):
        def residual_conv_block(x1, l2):
            filters = x1.shape[2]
            x2 = tf.keras.layers.Conv1D(
                filters,
                self.context_window,
                padding="same",
                activation="relu",
                kernel_regularizer=l2,
            )(x1)
            x2 = tf.keras.layers.Dropout(self.dropout)(x2)
            x2 = tf.keras.layers.LayerNormalization()(x2)
            return tf.keras.layers.add([x1, x2])

        def residual_attention(x1):
            x2 = HierarchicalAttention(self.attention_heads)(x1)
            x2 = tf.keras.layers.Dropout(self.dropout)(x2)
            x2 = tf.keras.layers.LayerNormalization()(x2)
            if self.attention_heads == 'same':
                return tf.keras.layers.add([x1, x2])
            else:
                return x2

        METRIC_DICT = {
            'precision': tf.keras.metrics.Precision(name='precision'),
            'recall': tf.keras.metrics.Recall(name='recall'),
            'f1': tfa.metrics.F1Score(
                nb_outputs, average='micro', threshold=self.threshold, name='f1'),
            'auc': tf.keras.metrics.AUC(name='auc')
        }

        embeddings_initializer = (
            tf.keras.initializers.Constant(embedding_matrix)
            if embedding_matrix
            else "uniform"
        )
        emb_dim = embedding_matrix.shape[1] if embedding_matrix else self.hidden_size

        l2 = tf.keras.regularizers.l2(self.l2)
        inp = tf.keras.layers.Input(shape=(sequence_length,))
        x = tf.keras.layers.Embedding(
            vocab_size,
            emb_dim,
            input_length=sequence_length,
            embeddings_initializer=embeddings_initializer,
        )(inp)
        x = tf.keras.layers.Dropout(
            self.dropout,
            noise_shape=(None, sequence_length, 1))(x)
        x = tf.keras.layers.LayerNormalization()(x)
        for i in range(self.nb_layers):
            x = residual_conv_block(x, l2)
        if self.attention:
            x = residual_attention(x)
        if self.feature_approach == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        elif self.feature_approach == 'sum':
            x = tf.keras.backend.sum(x, axis=1)
        elif self.feature_approach == 'concat':
            x = tf.keras.layers.Flatten()(x)
        else:
            raise NotImplementedError
        x = tf.keras.layers.Dense(
            self.dense_size, activation="relu", kernel_regularizer=l2
        )(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.LayerNormalization()(x)

        output_activation = (
            "sigmoid" if nb_outputs == 1 or self.multilabel else "softmax"
        )
        out = tf.keras.layers.Dense(
            nb_outputs,
            activation=output_activation,
            kernel_regularizer=l2,
        )(x)
        model = tf.keras.Model(inp, out)

        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            self.learning_rate, steps_per_epoch, self.learning_rate_decay,
            staircase=True
        )

        strategy = self._get_distributed_strategy()
        if isinstance(strategy, tf.distribute.MirroredStrategy):
            optimizer = tf.keras.optimizers.Adam(learning_rate)
        else:  # clipnorm is only supported in default strategy
            optimizer = tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0)
        metrics = [
            METRIC_DICT[m] if m in METRIC_DICT else m
            for m in self.metrics
        ]
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)
        return model

    def fit(self, X, Y=None, embedding_matrix=None, steps_per_epoch=None):
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(Y, list):
            Y = np.array(Y)

        if not (self.sequence_length and self.vocab_size and self.nb_outputs and steps_per_epoch):
            steps_per_epoch = self._init_from_data(X, Y)

        if isinstance(X, np.ndarray):
            data = self._prepare_data(X, Y, shuffle=True)
        else:  # tensorflow dataset
            data = X.batch(self.batch_size)

        train_steps_per_epoch = int((1-self.validation_split) * steps_per_epoch)
        if train_steps_per_epoch == 0:
            logger.warning(
                "Not enough data for validation. Consider decreasing \
                batch_size or validation_split. Some features that \
                rely on validation metrics like early stopping might \
                not work"
            )
        else:
            steps_per_epoch = train_steps_per_epoch
        train_data = data.take(steps_per_epoch)
        val_data = data.skip(steps_per_epoch)

        strategy = self._get_distributed_strategy()
        with strategy.scope():
            self.model = self._build_model(
                self.sequence_length, self.vocab_size, self.nb_outputs,
                steps_per_epoch, embedding_matrix)

        callbacks = []
        if self.tensorboard_log_path:
            datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=f"{self.tensorboard_log_path}/scalar/{datetime_str}"
            )
            callbacks.append(tensorboard)
        if self.early_stopping:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True)
            callbacks.append(early_stopping)

        self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.nb_epochs,
            callbacks=callbacks,
            verbose=self.verbose
        )
        return self

    def predict(self, X):
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(X, tf.data.Dataset):
            X = X.batch(self.batch_size)

        def yield_X_batch(X):
            if isinstance(X, np.ndarray):
                for i in range(0, X.shape[0], self.batch_size):
                    X_batch = X[i: i+self.batch_size]
                    yield X_batch
            else:  # tensorflow dataset
                yield from X

        if self.sparse_y:
            Y_pred = []
            for X_batch in yield_X_batch(X):
                Y_pred_batch = self.model.predict(X_batch) > self.threshold
                Y_pred.append(csr_matrix(Y_pred_batch))
            Y_pred = vstack(Y_pred)
            return Y_pred
        else:
            return self.model.predict(X) > self.threshold

    def predict_proba(self, X):
        # sparse_y not relevant as probs are dense
        return self.model.predict(X, self.batch_size)

    def score(self, X, Y):
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(Y, list):
            Y = np.array(Y)
        Y_pred = self.predict(X)
        return f1_score(Y_pred, Y, average="micro")

    def save(self, model_dir):
        self.model.save(model_dir)

    def load(self, model_dir):
        strategy = self._get_distributed_strategy()
        with strategy.scope():
            self.model = tf.keras.models.load_model(model_dir)
