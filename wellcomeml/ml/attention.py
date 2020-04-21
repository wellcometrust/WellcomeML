class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, attention_dim=20):
        super(SelfAttention, self).__init__()
        self.attention_dim = attention_dim

    def build(self, input_shape):
        self.WQ = self.add_weight(shape=(input_shape[-1], self.attention_dim), trainable=True, initializer='uniform')
        self.WK = self.add_weight(shape=(input_shape[-1], self.attention_dim), trainable=True, initializer='uniform')
        self.WV = self.add_weight(shape=(input_shape[-1], input_shape[-1]), trainable=True, initializer='uniform')

    def call(self, X):
    	"""
    	X: (batch_size, sequence_length, embedding_dimension)
    	"""
        Q = tf.matmul(X, self.WQ)
        K = tf.matmul(X, self.WK)
        V = tf.matmul(X, self.WV)
        
        attention_scores = tf.nn.softmax(tf.matmul(Q, tf.transpose(K, perm=[0,2,1])))
        return tf.matmul(attention_scores, V)

class AttentionMatrix(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionMatrix, self).__init__()
    
    def build(self, input_shape):
        self.attention_matrix = self.add_weight(shape=(input_shape[-2], input_shape[-2]), trainable=True, initializer='uniform')
    
    def call(self, X):
    	"""
    	X: (batch_size, sequence_length, embedding_dimension)
    	"""
        return tf.matmul(self.attention_matrix, X)