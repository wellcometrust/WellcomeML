from collections import defaultdict
import os

import tensorflow as tf
from transformers import (
    BertConfig,
    BertTokenizer,
    TFBertForSequenceClassification
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


class SemanticEquivalenceClassifier(BaseEstimator, TransformerMixin):
    """
    Class to fine-tune BERT-type models for semantic equivalence, for example
    paraphrase, textual similarity and other NLU tasks
    """
    def __init__(self, pretrained='bert',
                 batch_size=32,
                 eval_batch_size=32*2,
                 learning_rate=3e-5,
                 test_size=0.2,
                 max_length=128):
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
        if self.pretrained == 'bert':
            model_name = 'bert-base-cased'
            from_pt = False
        elif self.pretrained == 'scibert':
            model_name = 'allenai/scibert_scivocab_cased'
            from_pt = True

        self.config = BertConfig.from_pretrained(model_name, num_labels=2)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = TFBertForSequenceClassification.from_pretrained(
            model_name, config=self.config, from_pt=from_pt
        )

    def _prep_dataset_generator(self, X, y):
        batch_encoding = self.tokenizer.batch_encode_plus(
            X, max_length=self.max_length, add_special_tokens=True,
        )

        def gen_train():
            for i in range(len(X)):
                features = {k: pad(batch_encoding[k][i], self.max_length)
                            for k in batch_encoding}

                yield (features, int(y[i]))

        dataset = tf.data.Dataset.from_generator(
            gen_train,
            ({"input_ids": tf.int32,
              "attention_mask": tf.int32,
              "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

        return dataset

    def fit(self, X, y, random_state=None, epochs=3, **kwargs):
        """
        Fits a sentence similarity model

        Args:
            X: List of 2-uples: [('Sentence 1', 'Sentence 2'), ....]
            y: list of 0 (not similar) and 1s (similar)
            random_state: a random state for the train_test_split
            epochs: number_of epochs

        Returns:

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
            self.train_dataset = self.train_dataset.shuffle(self.max_length).\
                batch(self.batch_size).repeat(-1)
            self.valid_dataset = self.valid_dataset.batch(self.eval_batch_size)
    
            opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                           epsilon=1e-08)
    
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            )
    
            # Train and evaluate using tf.keras.Model.fit()
    
            metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
    
            self.model.compile(optimizer=opt, loss=loss, metrics=[metric])
    
            self.train_steps = len(X_train) // self.batch_size
            self.valid_steps = len(X_valid) // self.eval_batch_size
        finally:
            history = self.model.fit(
                self.train_dataset,
                epochs=epochs,
                steps_per_epoch=self.train_steps,
                validation_data=self.valid_dataset,
                validation_steps=self.valid_steps,
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
        X_tokenized = self.tokenizer.batch_encode_plus(
            X, max_length=self.max_length, add_special_tokens=True,
            pad_to_max_length=True,
            return_tensors="tf"
        )
        predictions = tf.convert_to_tensor(self.model.predict(X_tokenized))

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
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)

    def load(self, path):
        self._initialise_models()
        self.model = TFBertForSequenceClassification.from_pretrained(path)
        self.trained_ = True


def pad(x, pad_len):
    """Old versions of transformers do not pad by default"""
    return x + [0]*(pad_len-len(x))
