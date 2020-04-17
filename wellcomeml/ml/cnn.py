"""
CNN architecture inspired from spacy for NLP tasks

It follows the embed, encode, attend, predict framework
Embed: learns a new embedding but can receive
    pre trained embeddings as well
Encode: stacked CNN with context window 3 that maintains
    the size of input and applies dropout and layer norm
Attend: not yet implemented
Predict: softmax or sigmoid depending on number of outputs
    and whether task is multilabel
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
import tensorflow as tf


class CNN(BaseEstimator, ClassifierMixin):
    def __init__(self, context_window = 3, learning_rate=0.001, batch_size=32, nb_epochs=5, dropout=0.2, multilabel=False):
        self.context_window = context_window
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.dropout = dropout
        self.multilabel = multilabel

    def fit(self, X, Y, embedding_matrix=None):
        sequence_length = X.shape[1]
        vocab_size = X.max() + 1
        emb_dim = embedding_matrix.shape[1] if embedding_matrix else 100
        nb_outputs = max(Y)

        def residual_conv_block(x1):
            filters = x1.shape[2]
            x2 = tf.keras.layers.Conv1D(filters, self.context_window, padding='same', activation='relu')(x1)
            x2 = tf.keras.layers.Dropout(self.dropout)(x2)
            x2 = tf.keras.layers.LayerNormalization()(x2)
            return tf.keras.layers.add([x1, x2])

        embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix) if embedding_matrix else 'uniform'
        inp = tf.keras.layers.Input(shape=(sequence_length,))
        x = tf.keras.layers.Embedding(
                vocab_size,
                emb_dim,
                input_length=sequence_length,
                embeddings_initializer=embeddings_initializer)(inp)
        x = tf.keras.layers.LayerNormalization()(x)
        x = residual_conv_block(x)
        x = residual_conv_block(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.LayerNormalization()(x)

        output_activation = 'sigmoid' if nb_outputs==1 or self.multilabel else 'softmax'
        out = tf.keras.layers.Dense(nb_outputs, activation=output_activation)(x)

        self.model = tf.keras.Model(inp, out)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y, epochs=self.nb_epochs, batch_size=self.batch_size, validation_split=0.1)
        return self

    def predict(self, X):
        return self.model(X).numpy() > 0.5

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return f1_score(Y_pred, Y, average='micro')
