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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf

from wellcomeml.ml.attention import HierarchicalAttention
from wellcomeml.ml.keras_utils import Metrics

class CNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, context_window = 3, learning_rate=0.001,
                 batch_size=32, nb_epochs=5, dropout=0.2,
                 nb_layers=4, hidden_size=100, multilabel=False,
                 attention=False):
        self.context_window = context_window
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.dropout = dropout
        self.nb_layers = nb_layers
        self.hidden_size = hidden_size # note that on current implementation CNN use same hidden size as embedding so if embedding matrix is passed, this is not used. in the future we can decouple
        self.multilabel = multilabel
        self.attention = attention

    def fit(self, X, Y, embedding_matrix=None):
        sequence_length = X.shape[1]
        vocab_size = X.max() + 1
        emb_dim = embedding_matrix.shape[1] if embedding_matrix else self.hidden_size
        nb_outputs = Y.max() if not self.multilabel else Y.shape[1]

        def residual_conv_block(x1):
            filters = x1.shape[2]
            x2 = tf.keras.layers.Conv1D(
                filters,
                self.context_window,
                padding='same',
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(1e-6)
            )(x1)
            x2 = tf.keras.layers.Dropout(self.dropout)(x2)
            x2 = tf.keras.layers.LayerNormalization()(x2)
            return tf.keras.layers.add([x1, x2])

        def residual_attention(x1):
            x2 = HierarchicalAttention()(x1)
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
        for i in range(self.nb_layers):
            x = residual_conv_block(x)
        if self.attention:
            x = residual_attention(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-6))(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.LayerNormalization()(x)

        output_activation = 'sigmoid' if nb_outputs==1 or self.multilabel else 'softmax'
        out = tf.keras.layers.Dense(nb_outputs, activation=output_activation, kernel_regularizer=tf.keras.regularizers.l2(1e-6))(x)
        self.model = tf.keras.Model(inp, out)

        optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[])

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, shuffle= True)
        metrics = Metrics(validation_data=(X_val, Y_val))
        self.model.fit(X_train, Y_train, epochs=self.nb_epochs, batch_size=self.batch_size, validation_data=(X_val, Y_val), callbacks=[metrics])
        return self

    def predict(self, X):
        return self.model(X).numpy() > 0.5

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return f1_score(Y_pred, Y, average='micro')
