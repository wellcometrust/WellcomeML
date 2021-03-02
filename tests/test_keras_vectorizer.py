import tempfile
import os

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


def test_build_embedding_matrix():

    X = ["One", "Two", "Three"]

    vocab_size = 1
    keras_vectorizer = KerasVectorizer(vocab_size=vocab_size)
    keras_vectorizer.fit(X)

    with tempfile.TemporaryDirectory() as tmp_dir:
        embeddings_path = os.path.join(tmp_dir, "embeddings.csv")
        embeddings = [
            "one 0 1 0 0 0",
            "two 0 0 1 0 0",
            "three 0 0 0 1 0",
            "four 0 0 0 0 1",
        ]
        with open(embeddings_path, "w") as embeddings_path_tmp:
            for line in embeddings:
                embeddings_path_tmp.write(line)
                embeddings_path_tmp.write("\n")
        embedding_matrix = keras_vectorizer.build_embedding_matrix(
            embeddings_name_or_path=embeddings_path
        )

        assert embedding_matrix.shape == (5, 5)


def test_build_embedding_matrix_word_vectors():

    X = ["One", "Two", "Three"]

    vocab_size = 1
    keras_vectorizer = KerasVectorizer(vocab_size=vocab_size)
    keras_vectorizer.fit(X)

    embedding_matrix = keras_vectorizer.build_embedding_matrix(
        embeddings_name_or_path="glove-twitter-25"
    )

    assert embedding_matrix.shape == (5, 25)


def test_infer_from_data():
    X = ["One", "Two words", "Three words here"]

    keras_vectorizer = KerasVectorizer()
    keras_vectorizer.fit(X)

    assert keras_vectorizer.sequence_length == 3
