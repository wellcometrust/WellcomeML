from collections import defaultdict
from datetime import datetime
import math
import os

from transformers import BertConfig, BertTokenizer, \
    TFBertForSequenceClassification

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from wellcomeml.ml.keras_utils import CategoricalMetrics  # , MetricMiniBatchHistory
from wellcomeml.logger import LOGGING_LEVEL, build_logger
from wellcomeml.utils import throw_extra_import_message

try:
    import tensorflow as tf
except ImportError as e:
    throw_extra_import_message(error=e, required_module='tensorflow', extra='tensorflow')


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
        random_seed=42,
        buffer_size=100,
        logging_level=LOGGING_LEVEL,
        tensorflow_log_path="logs",
        verbose=1  # follows Keras verbose for now
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
            random_seed: A seed used for shuffling the dataset upon training
            buffer_size: A buffer size that will be used when shuffling the dataset
            tensorflow_log_path: Path to store tensorboard logs
            verbose: 0,1,2. Verbose level for keras fit
        """
        self.logger = build_logger(logging_level, __name__)
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.test_size = test_size
        self.max_length = max_length
        self.random_seed = random_seed
        self.buffer_size = buffer_size
        self.tensorboard_log_path = tensorflow_log_path
        self.verbose = verbose

        # Defines attributes that will be initialised later
        self.config = None
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.valid_dataset = None
        self.train_steps = None
        self.valid_steps = None
        self.strategy = None

        self.history = defaultdict(list)

    # Bert models have a tensorflow checkopoint, otherwise,
    # we need to load the pytorch versions with the parameter `from_pt=True`

    def initialise_models(self):
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

    def _prep_dataset_generator(self, X, y=None, shuffle=False, repeat=False, batch_size=32):
        features = ["input_ids", "attention_mask", "token_type_ids"]

        if y is None:
            input_element_types = {feature: tf.int32 for feature in features}
            input_element_tensors = (
                {feature: tf.TensorShape([None]) for feature in features}
            )

        else:
            input_element_types = ({feature: tf.int32 for feature in features}, tf.int64)
            input_element_tensors = (
                {feature: tf.TensorShape([None]) for feature in features},
                tf.TensorShape([]),
            )

        self.logger.info("Tokenising texts.")
        batch_encoding = self.tokenizer.batch_encode_plus(
            X, max_length=self.max_length, add_special_tokens=True,
        )
        self.logger.info("Configuring dataset generators.")

        def gen_train():
            for i in range(len(X)):
                features_dict = {
                    k: pad(batch_encoding[k][i], self.max_length)
                    for k in batch_encoding
                }
                if y is None:
                    yield features_dict
                else:
                    yield (features_dict, int(y[i]))

        dataset = tf.data.Dataset.from_generator(
            gen_train, input_element_types, input_element_tensors,
        )

        if shuffle:
            dataset = dataset.shuffle(self.buffer_size, seed=self.random_seed)

        dataset = dataset.batch(batch_size)
        if repeat:
            dataset = dataset.repeat(-1)

        return dataset

    def _compile_model(self, metrics=[]):
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                       epsilon=1e-08)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        precision = CategoricalMetrics(metric="precision")
        recall = CategoricalMetrics(metric="recall")
        f1_score = CategoricalMetrics(metric="f1_score")
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

        metrics = [accuracy, precision, recall, f1_score] + metrics

        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)

    def _get_distributed_strategy(self):
        if len(tf.config.list_physical_devices("GPU")) > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.get_strategy()
        return strategy

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
            self.strategy = self._get_distributed_strategy()
            with self.strategy.scope():
                self.initialise_models()

            # Train/val split
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=self.test_size, random_state=random_state
            )

            # Generates tensorflow dataset

            self.train_dataset = self._prep_dataset_generator(X_train, y_train,
                                                              shuffle=True,
                                                              repeat=True,
                                                              batch_size=self.batch_size)
            self.valid_dataset = self._prep_dataset_generator(X_valid, y_valid,
                                                              batch_size=self.eval_batch_size)

            with self.strategy.scope():
                self._compile_model()

            self.train_steps = math.ceil(len(X_train)/self.batch_size)
            self.valid_steps = math.ceil(len(X_valid)/self.eval_batch_size)
        finally:
            callback_objs = [
                # Issue #187
                # MetricMiniBatchHistory()
            ]
            if self.tensorboard_log_path:
                datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=f"{self.tensorboard_log_path}/scalar/{datetime_str}"
                )
                callback_objs.append(tensorboard)
            self.logger.info("Fitting model")
            history = self.model.fit(
                self.train_dataset,
                epochs=epochs,
                steps_per_epoch=self.train_steps,
                validation_steps=self.valid_steps,
                validation_data=self.valid_dataset,
                callbacks=callback_objs,
                verbose=self.verbose,
                **kwargs
            )

            # Accumulates history of metrics from different partial fits
            for metric, values in history.history.items():
                self.history[metric] += values

            self.trained_ = True

        return self

    def predict_proba(self, X):
        """
        Calculates scores for model prediction

        Args:
            X: List of 2-uples: [('Sentence 1', 'Sentence 2'), ....]

        Returns:
            An array of shape len(X) x 2 with scores for classes 0 and 1

        """
        # I didn't quite get to the bottom of this error, but with mirrored strategy predicting
        # I need to predict "manually" by calling the model.
        # Any progress on this can be tracked on #241

        if isinstance(self.strategy, tf.distribute.MirroredStrategy):
            predictions = []
            for batch in self._prep_dataset_generator(X):
                predictions += [self.model(batch).logits]

            predictions = tf.concat(predictions, axis=0)
        else:
            predictions = self.model.predict(self._prep_dataset_generator(X)).logits

        return tf.nn.softmax(predictions).numpy()

    def predict(self, X):
        """
        Calculates the predicted class (0 - for negative and 1 for positive)
        for X.

        Args:
            X: List of 2-uples: [('Sentence 1', 'Sentence 2'), ....]

        Returns:
            An array of 0s and 1s

        """
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path):
        """Saves model to path"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)

    def load(self, path):
        """Loads model from path"""
        self.strategy = self._get_distributed_strategy()

        with self.strategy.scope():
            self.initialise_models()
            self.model = TFBertForSequenceClassification.from_pretrained(path)
        self.trained_ = True


class SemanticEquivalenceMetaClassifier(SemanticEquivalenceClassifier):
    """
    Class to fine tune Semantic Classifier, allowing the possibility of
    adding metadata to the classification. Extends
    SemanticEquivalenceClassifier, and extends modules for data prep
    and model initialisation. Fit, predict, predict_proba remain intact
    """

    def __init__(self, n_numerical_features, dropout_rate=0,
                 batch_norm=False,  **kwargs):
        """
        Initialises SemanticEquivalenceMetaClassifier, with n_numerical_features

        Args:
            n_numerical_features(int): Number of features of the model which
            are not text
            dropout_rate(float): Dropout rate after concatenating features
            (default = 0, i.e. no dropout)
            batch_norm(bool): Whether to apply batch normalisation after the
            last dense layer
            **kwargs: Any kwarg to `SemanticEquivalenceClassifier`
        """
        super().__init__(**kwargs)
        self.n_numerical_features = n_numerical_features
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.model = None

    def _separate_features(self, X):
        X_text = [x[:2] for x in X]
        X_numerical = [x[2: 2 + self.n_numerical_features] for x in X]

        return X_text, X_numerical

    def initialise_models(self):
        """Extends/overrides super class initialisation of model"""
        super_model = super().initialise_models()

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

        # Drop out layer to the Bert features if rate > 0
        x = (tf.keras.layers.Dropout(self.dropout_rate)(x)
             if self.dropout_rate > 0 else x)

        # Concatenates with numerical features
        x = tf.keras.layers.concatenate(
            [x, input_numerical_data_tensor[0]], name="concatenate"
        )

        # Batch norm to the concatenated layer if self.batch_norm=True
        x = (tf.keras.layers.BatchNormalization(name="batch_norm")(x)
             if self.batch_norm else x)

        # Dense layer that will be used for softmax prediction later
        x = tf.keras.layers.Dense(2, name="dense_layer")(x)

        self.model = tf.keras.Model(input_text_tensors +
                                    input_numerical_data_tensor, x)

    def _prep_dataset_generator(self, X, y=None, shuffle=False, repeat=False, batch_size=32):
        """Overrides/extends the super class data preparation"""
        text_features = ["input_ids", "attention_mask", "token_type_ids"]
        X_text, X_numerical = self._separate_features(X)

        batch_encoding_text = self.tokenizer.batch_encode_plus(
            X_text, max_length=self.max_length, add_special_tokens=True,
        )

        if y is None:
            input_element_types = (
                {
                    **{feature: tf.int32 for feature in text_features},
                    **{"numerical_metadata": tf.float32},
                }
            )
            input_element_tensors = (
                {
                    **{feature: tf.TensorShape([None]) for feature in text_features},
                    **{"numerical_metadata": tf.TensorShape([self.n_numerical_features])},
                }
            )
        else:
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

        def gen_train():
            for i in range(len(X)):
                features_dict = {
                    k: pad(batch_encoding_text[k][i], self.max_length)
                    for k in batch_encoding_text
                }
                features_dict["numerical_metadata"] = X_numerical[i]

                if y is None:
                    yield features_dict
                else:
                    yield (features_dict, int(y[i]))

        dataset = tf.data.Dataset.from_generator(
            gen_train, input_element_types, input_element_tensors,
        )

        if shuffle:
            dataset = dataset.shuffle(self.buffer_size, seed=self.random_seed)

        dataset = dataset.batch(batch_size)
        if repeat:
            dataset = dataset.repeat(-1)

        return dataset

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

    def predict_proba(self, X):
        """
        Calculates scores for metadata semantic classifier.

        Args:
            X(list of t-uples): Assumes that each element x of X has length
            2+n_numerical_features, the first two components are texts and
            remaining components are numerical features (e.g.)

        Returns:
            Numpy array of probabilities, of shape len(X) x 2

        """
        predictions = self.model.predict(self._prep_dataset_generator(X))

        return tf.nn.softmax(predictions).numpy()

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
        strategy = self._get_distributed_strategy()
        with strategy.scope():
            self.initialise_models()
            self.model = tf.keras.models.load_model(path)
        self.trained_ = True


def pad(x, pad_len):
    """Old versions of transformers do not pad by default"""
    return x + [0] * (pad_len - len(x))
