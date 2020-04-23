from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
import tensorflow as tf

from wellcomeml.ml.keras_utils import Metrics

class BiLSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, batch_size=32, nb_epochs=5,
                 dropout=0.1, nb_layers=2, multilabel=False):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.dropout = dropout
        self.nb_layers = nb_layers
        self.multilabel = multilabel

    def fit(self, X, Y, embedding_matrix=None, *_):
        sequence_length = X.shape[1]
        vocab_size = X.max() + 1
        embedding_size = embedding_matrix.shape[1] if embedding_matrix else 100
                                 
        nb_outputs = Y.max() if not self.multilabel else Y.shape[1]
        output_activation = 'sigmoid' if nb_outputs==1 or self.multilabel else 'softmax'

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, shuffle=True)
        
        l2 = tf.keras.regularizers.l2(1e-6)
        embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix) if embedding_matrix else 'uniform'
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                vocab_size,
                embedding_size,
                input_length=sequence_length,
                embeddings_initializer=embeddings_initializer
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=l2)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, kernel_regularizer=l2)),
            tf.keras.layers.Dense(20, kernel_regularizer=l2),
            tf.keras.layers.Dense(nb_outputs, activation=output_activation, kernel_regularizer=l2)
        ])
        optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate, clipnorm=1.0)
        metrics = Metrics(validation_data=(X_val, Y_val))
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[])
        self.model.fit(X, Y, epochs=self.nb_epochs, batch_size=self.batch_size, validation_data=(X_val, Y_val), callbacks=[metrics])
        return self

    def predict(self, X, *_):
        return self.model(X).numpy() > 0.5

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return f1_score(Y, Y_pred, average='micro')
