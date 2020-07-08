from collections import defaultdict
import os

import tensorflow as tf
from transformers import BertConfig, BertTokenizer, TFBertForSequenceClassification

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from wellcomeml.ml.keras_utils import CategoricalMetrics, MetricMiniBatchHistory


class SemanticEquivalenceClassifier(BaseEstimator, TransformerMixin):
    """
    Class to fine-tune BERT-type models for semantic equivalence, for example
    paraphrase, textual similarity and other NLU tasks
    """

    def __init__(
        self,
        pretrained="bert",
        batch_size=32,
        eval_batch_size=32 * 2,
        learning_rate=3e-5,
        test_size=0.2,
        max_length=128,
    ):
        """

        Args:
            pretrained(str): 'bert' or 'scibert'
            batch_size(int): The batch size for training
            eval_batch_size(int): The evaluation batch size
            learning_rate(float): Learning rate (usually between 0.1 and 1e-5)
            test_size(float): Proportion of data used for testing (default 0.2)
            max_length: Maximum length of text in characters.
             Zero pads every text smaller than this number and cuts out
             any text bigger than that number
        """
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.test_size = test_size
        self.max_length = max_length

        # Defines attributes that will be initialised later
        self.config = None
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.valid_dataset = None
        self.train_steps = None
        self.valid_steps = None

        self.history = defaultdict(list)

    # Bert models have a tensorflow checkopoint, otherwise,
    # we need to load the pytorch versions with the parameter `from_pt=True`

    def _initialise_models(self):
        if self.pretrained == "bert":
            model_name = "bert-base-cased"
            from_pt = False
        elif self.pretrained == "scibert":
            model_name = "allenai/scibert_scivocab_cased"
            from_pt = True

        self.config = BertConfig.from_pretrained(model_name, num_labels=2)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = TFBertForSequenceClassification.from_pretrained(
            model_name, config=self.config, from_pt=from_pt
        )
        return self.model

    def _prep_dataset_generator(self, X, y):
        features = ["input_ids", "attention_mask", "token_type_ids"]

        batch_encoding = self.tokenizer.batch_encode_plus(
            X, max_length=self.max_length, add_special_tokens=True,
        )

        def gen_train():
            for i in range(len(X)):
                features = {
                    k: pad(batch_encoding[k][i], self.max_length)
                    for k in batch_encoding
                }

                yield (features, int(y[i]))

        input_element_types = ({feature: tf.int32 for feature in features}, tf.int64)
        input_element_tensors = (
            {feature: tf.TensorShape([None]) for feature in features},
            tf.TensorShape([]),
        )

        dataset = tf.data.Dataset.from_generator(
            gen_train, input_element_types, input_element_tensors,
        )

        return dataset

    def _compile_model(self, metrics=[]):
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=1e-08)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        precision = CategoricalMetrics(metric="precision")
        recall = CategoricalMetrics(metric="recall")
        f1_score = CategoricalMetrics(metric="f1_score")
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

        metrics = [accuracy, precision, recall, f1_score] + metrics

        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)

    def _tokenize(self, X):
        return self.tokenizer.batch_encode_plus(
            X,
            max_length=self.max_length,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

    def _prep_data_for_prediction(self, X):
        X_tokenized = self._tokenize(X)
        return tf.convert_to_tensor(self.model.predict(X_tokenized))

    def fit(self, X, y, random_state=None, epochs=3, metrics=[], **kwargs):
        """
        Fits a sentence similarity model

        Args:
            X: List of 2-uples: [('Sentence 1', 'Sentence 2'), ....]
            y: list of 0 (not similar) and 1s (similar)
            random_state: a random state for the train_test_split
            epochs: number_of epochs

        Returns:
            A fitted model

        """
        # Initialises/downloads model if not trained before.
        # If trained, fits extra epochs without the transformations
        try:
            check_is_fitted(self)
        except NotFittedError:
            self._initialise_models()

            # Train/val split
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=self.test_size, random_state=random_state
            )

            # Generates tensorflow dataset
            self.train_dataset = self._prep_dataset_generator(X_train, y_train)
            self.valid_dataset = self._prep_dataset_generator(X_valid, y_valid)

            # Generates mini-batches and stores in a class variable
            self.train_dataset = (
                self.train_dataset.shuffle(self.max_length)
                .batch(self.batch_size)
                .repeat(-1)
            )
            self.valid_dataset = self.valid_dataset.batch(self.eval_batch_size)

            self._compile_model()

            self.train_steps = len(X_train) // self.batch_size
            self.valid_steps = len(X_valid) // self.eval_batch_size
        finally:
            history = self.model.fit(
                self.train_dataset,
                epochs=epochs,
                steps_per_epoch=self.train_steps,
                validation_data=self.valid_dataset,
                validation_steps=self.valid_steps,
                callbacks=[MetricMiniBatchHistory()],
                **kwargs
            )

            # Accumulates history of metrics from different partial fits
            for metric, values in history.history.items():
                self.history[metric] += values

            self.trained_ = True

        return self

    def score(self, X):
        """
        Calculates scores for model prediction

        Args:
            X: List of 2-uples: [('Sentence 1', 'Sentence 2'), ....]

        Returns:
            An array of shape len(X) x 2 with scores for classes 0 and 1

        """
        predictions = self._prep_data_for_prediction(X)

        return tf.keras.activations.softmax(predictions).numpy()

    def predict(self, X):
        """
        Calculates the predicted class (0 - for negative and 1 for positive)
        for X.

        Args:
            X: List of 2-uples: [('Sentence 1', 'Sentence 2'), ....]

        Returns:
            An array of 0s and 1s

        """
        return self.score(X).argmax(axis=1)

    def save(self, path):
        """Saves model to path"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)

    def load(self, path):
        """Loads model from path"""
        self._initialise_models()
        self.model = TFBertForSequenceClassification.from_pretrained(path)
        self.trained_ = True


class SemanticEquivalenceMetaClassifier(SemanticEquivalenceClassifier):
    """
    Class to fine tune Semantic Classifier, allowing the possibility of
    adding metadata to the classification. Extends
    SemanticEquivalenceClassifier, and extends modules for data prep
    and model initialisation. Fit, predict, score remain intact
    """

    def __init__(self, n_numerical_features, dropout_rate=0.1, **kwargs):
        """
        Initialises SemanticEquivalenceMetaClassifier, with n_numerical_features

        Args:
            n_numerical_features(int): Number of features of the model which
            are not text
            dropout_rate(float): Dropout rate after concatenating features
            (default = 0.1)
            **kwargs: Any kwarg to `SemanticEquivalenceClassifier`
        """
        super().__init__(**kwargs)
        self.n_numerical_features = n_numerical_features
        self.dropout_rate = dropout_rate
        self.model = None

    def _separate_features(self, X):
        X_text = [x[:2] for x in X]
        X_numerical = [x[2: 2 + self.n_numerical_features] for x in X]

        return X_text, X_numerical

    def _initialise_models(self):
        """Extends/overrides super class initialisation of model"""
        super_model = super()._initialise_models()

        # Define text input features
        text_features = ["input_ids", "attention_mask", "token_type_ids"]
        input_text_tensors = [
            tf.keras.layers.Input(
                name=feature_name, shape=tf.TensorShape([None]), dtype=tf.int32
            )
            for feature_name in text_features
        ]

        input_numerical_data_tensor = [
            tf.keras.layers.Input(
                name="numerical_metadata",
                shape=tf.TensorShape([self.n_numerical_features]),
                dtype=tf.float32,
            )
        ]
        # Calls the CLS layer of Bert
        x = super_model.bert(input_text_tensors)[1]

        # Drop out layer to the Bert features
        x = super_model.dropout(x, training=False)

        # Concatenates with numerical features
        x = tf.keras.layers.concatenate(
            [x, input_numerical_data_tensor[0]], name="concatenate"
        )

        # Dense layer that will be used for softmax prediction later
        x = tf.keras.layers.Dense(2, name="dense_layer")(x)

        self.model = tf.keras.Model(input_text_tensors + input_numerical_data_tensor, x)

    def _prep_dataset_generator(self, X, y):
        """Overrides/extends the super class data preparation"""
        text_features = ["input_ids", "attention_mask", "token_type_ids"]
        X_text, X_numerical = self._separate_features(X)

        batch_encoding_text = self.tokenizer.batch_encode_plus(
            X_text, max_length=self.max_length, add_special_tokens=True,
        )

        def gen_train():
            for i in range(len(X)):
                features = {
                    k: pad(batch_encoding_text[k][i], self.max_length)
                    for k in batch_encoding_text
                }
                features["numerical_metadata"] = X_numerical[i]

                yield (features, int(y[i]))

        input_element_types = (
            {
                **{feature: tf.int32 for feature in text_features},
                **{"numerical_metadata": tf.float32},
            },
            tf.int64,
        )
        input_element_tensors = (
            {
                **{feature: tf.TensorShape([None]) for feature in text_features},
                **{"numerical_metadata": tf.TensorShape([self.n_numerical_features])},
            },
            tf.TensorShape([]),
        )

        dataset = tf.data.Dataset.from_generator(
            gen_train, input_element_types, input_element_tensors,
        )

        return dataset

    def _prep_data_for_prediction(self, X):
        X_text, X_numerical = self._separate_features(X)

        X_processed = self._tokenize(X_text)
        X_processed["numerical_metadata"] = tf.convert_to_tensor(X_numerical)

        predictions = tf.convert_to_tensor(self.model.predict(X_processed))
        return predictions

    def fit(self, X, y, **kwargs):
        """
        Fits semantic classifier

        Args:
            X(list of t-uples): Assumes that each element x of X has length
            2+n_numerical_features, the first two components are texts and
            remaining components are numerical features (e.g.)
            y: list of 0 (not similar) and 1s (similar)
            random_state: a random state for the train_test_split
            epochs: number_of epochs

        Returns:
             Array of probabilities, of shape len(X) x 2

        """
        return super().fit(X, y, **kwargs)

    def score(self, X):
        """
        Calculates scores for metadata semantic classifier.

        Args:
            X(list of t-uples): Assumes that each element x of X has length
            2+n_numerical_features, the first two components are texts and
            remaining components are numerical features (e.g.)

        Returns:
            Numpy array of probabilities, of shape len(X) x 2

        """

        return super().score(X)

    def predict(self, X):
        """
        Calculates scores for metadata semantic classifier.

        Args:
            X(list of t-uples): Assumes that each element x of X has length
            2+n_numerical_features, the first two components are texts and
            remaining components are numerical features (e.g.)

        Returns:
            Array of predictions of length len(X)

        """
        return super().predict(X)

    def save(self, path):
        """Saves meta model to path"""
        self.model.save(path)

    def load(self, path):
        """Loads metamodel from path"""
        self._initialise_models()
        self.model = tf.keras.models.load_model(path)
        self.trained_ = True


def pad(x, pad_len):
    """Old versions of transformers do not pad by default"""
    return x + [0] * (pad_len - len(x))
