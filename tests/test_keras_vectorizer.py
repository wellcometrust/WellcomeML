from wellcomeml.ml import KerasVectorizer


def test_vanilla():
    X = ["One", "Two", "Three Four"]

    keras_vectorizer = KerasVectorizer()
    X_vec = keras_vectorizer.fit_transform(X)

    assert X_vec.shape[0] == 3
    assert X_vec.shape[1] == 2
    assert X_vec.max() == 5  # 4 tokens including OOV


def test_sequence_length():
    X = ["One", "Two", "Three"]

    sequence_length = 5
    keras_vectorizer = KerasVectorizer(sequence_length=sequence_length)
    X_vec = keras_vectorizer.fit_transform(X)

    assert X_vec.shape[1] == sequence_length


def test_vocab_size():
    X = ["One", "Two", "Three"]

    vocab_size = 1
    keras_vectorizer = KerasVectorizer(vocab_size=vocab_size)
    X_vec = keras_vectorizer.fit_transform(X)

    assert X_vec.max() == vocab_size
