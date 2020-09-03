import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf"""

    def __init__(self, attention_dim=20):
        super(SelfAttention, self).__init__()
        self.attention_dim = attention_dim

    def build(self, input_shape):
        self.WQ = self.add_weight(
            shape=(input_shape[-1], self.attention_dim),
            trainable=True,
            initializer="uniform",
        )
        self.WK = self.add_weight(
            shape=(input_shape[-1], self.attention_dim),
            trainable=True,
            initializer="uniform",
        )
        self.WV = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            trainable=True,
            initializer="uniform",
        )

    def call(self, X):
        """
        In: (batch_size, sequence_length, embedding_dimension)
        Out: (batch_size, sequence_length, embedding_dimension)
        """
        Q = tf.matmul(X, self.WQ)
        K = tf.matmul(X, self.WK)
        V = tf.matmul(X, self.WV)

        attention_scores = tf.nn.softmax(tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1])))
        return tf.matmul(attention_scores, V)


class FeedForwardAttention(tf.keras.layers.Layer):
    """https://colinraffel.com/publications/iclr2016feed.pdf"""

    def __init__(self):
        super(FeedForwardAttention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1), trainable=True, initializer="uniform"
        )

    def call(self, X):
        """
        In: (batch_size, sequence_length, embedding_dimension)
        Out: (batch_size, embedding_dimension)
        """
        e = tf.math.tanh(tf.matmul(X, self.W))
        attention_scores = tf.nn.softmax(e)
        return tf.matmul(tf.transpose(X, perm=[0, 2, 1]), attention_scores)


class HierarchicalAttention(tf.keras.layers.Layer):
    """https://www.aclweb.org/anthology/N16-1174/"""

    def __init__(self, attention_heads='same'):
        super(HierarchicalAttention, self).__init__()
        self.attention_heads = attention_heads

    def build(self, input_shape):
        if self.attention_heads == 'same':
            nb_attention_heads = input_shape[-2]
        else:
            nb_attention_heads = self.attention_heads
        self.attention_matrix = self.add_weight(
            shape=(input_shape[-1], nb_attention_heads),
            trainable=True,
            initializer="uniform",
        )

    def call(self, X):
        """
        In: (batch_size, sequence_length, embedding_dimension)
        Out: (batch_size, sequence_length, embedding_dimension)
        """
        attention_scores = tf.nn.softmax(
            tf.math.tanh(tf.matmul(X, self.attention_matrix))
        )
        return tf.matmul(tf.transpose(attention_scores, perm=[0, 2, 1]), X)
