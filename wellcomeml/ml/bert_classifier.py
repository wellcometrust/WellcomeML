"""
Implements BertClassifier, an sklearn compatible BERT
classifier class that can be used for text classification
tasks
"""
import math
import os

from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from wellcomeml.utils import throw_extra_import_message

try:
    import tensorflow as tf
except ImportError as e:
    throw_extra_import_message(error=e, required_module='tensorflow', extra='tensorflow')


PRETRAINED_CONFIG = {
    "bert": {
        "name": "bert-base-uncased",
        "from_pt": False
    },
    "scibert": {
        "name": "allenai/scibert_scivocab_uncased",
        "from_pt": True
    }
}


class BertClassifier(BaseEstimator, TransformerMixin):
    """
    Class to fine-tune BERT like models for a text
    classification task
    """
    def __init__(self, learning_rate=5e-5, epochs=5, batch_size=8,
                 pretrained="bert-base-uncased", threshold=0.5,
                 validation_split=0.1, max_length=512, multilabel=True,
                 from_pt=False, random_seed=42):
        """
        Args:
           learning_rate(float): learning rate to optimize model, default 5e-5
           epochs(int): number of epochs to train the model, default 5
           batch_size(int): batch size to be used in training, default 8
           pretrained(str): bert, scibert or any transformers compatible, default bert-base-uncased
           threshold(float): threshold upon which to assign class, default 0.5
           validation_split(float): split for validation set during training, default 0.1
           max_length(int): maximum number of tokens, controls padding and truncation, default 512
           multilabel(bool): flag on whether the problem is multilabel i.e. Y a matrix or a column
           random_seed(int): controls random seed for reproducibility
           from_pt(bool): flag about pretrained model in pytorch or tensorflow
        """
        self.learning_rate = learning_rate
        self.epochs = epochs  # used to be n_iterations
        self.batch_size = batch_size
        self.pretrained = pretrained
        self.threshold = threshold
        self.validation_split = validation_split
        self.max_length = max_length
        self.multilabel = multilabel
        self.random_seed = random_seed
        self.from_pt = from_pt
        self.initiated_ = False

    def _init_model(self, num_labels=2):
        config = {
            "name": self.pretrained,
            "from_pt": self.from_pt
        }
        if self.pretrained in PRETRAINED_CONFIG:
            config = PRETRAINED_CONFIG[self.pretrained]

        pretrained = config["name"]
        from_pt = config["from_pt"]

        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.model = TFBertForSequenceClassification.from_pretrained(
            pretrained, from_pt=from_pt, num_labels=num_labels)
        self.initiated_ = True

    def _transform_data(self, input_ids, attention_mask, labels=None):
        input_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        if labels is not None:
            return input_data, labels
        else:
            return input_data

    def _prepare_data(self, X, Y=None, shuffle=False, repeat=False):
        X_vec = self.tokenizer.batch_encode_plus(
            X, max_length=self.max_length, pad_to_max_length=True,
            add_special_tokens=True, return_token_type_ids=False
        )

        if Y is not None:
            dataset = tf.data.Dataset.from_tensor_slices(
                (X_vec["input_ids"], X_vec["attention_mask"], Y)
            )
        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                (X_vec["input_ids"], X_vec["attention_mask"])
            )
        dataset = dataset.map(self._transform_data)
        if shuffle:
            dataset = dataset.shuffle(100, seed=self.random_seed)
        dataset = dataset.batch(self.batch_size)
        if repeat:
            dataset = dataset.repeat(self.epochs)
        return dataset

    def fit(self, X, Y):
        """
        Fine tune BERT to text data and Y vector

        Args:
           X: list or numpy array of texts (n_samples,)
           Y: list or numpy array of classes (n_samples, n_classes)
        """
        num_labels = len(Y[0])
        if not self.initiated_:
            self._init_model(num_labels)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, random_state=self.random_seed, test_size=self.validation_split
        )

        train_data = self._prepare_data(X_train, Y_train, shuffle=True, repeat=True)
        val_data = self._prepare_data(X_test, Y_test)

        steps_per_epoch = math.ceil(len(X_train) / self.batch_size)
        val_steps_per_epoch = math.ceil(len(X_test) / self.batch_size)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        if self.multilabel:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        #  TODO: Fix error when adding metrics
        #  metrics = [tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        self.model.compile(optimizer=optimizer, loss=loss)

        history = self.model.fit(
            train_data, validation_data=val_data, epochs=self.epochs,
            steps_per_epoch=steps_per_epoch, validation_steps=val_steps_per_epoch)
        self.losses = history.history["loss"]

        return self

    def _predict_proba(self, X):
        dataset = self._prepare_data(X)
        out = self.model.predict(dataset)[0]
        if self.multilabel:
            return tf.nn.sigmoid(out).numpy()
        else:
            return tf.nn.softmax(out).numpy()

    def predict(self, X):
        """
        Predict vector Y on text data X

        Args:
           X: list or numpy array of texts (n_samples)
        Returns:
           Y_pred: numpy array of predictions Y (n_samples, n_classes)
        """
        Y_pred_proba = self._predict_proba(X)
        if self.multilabel:
            return Y_pred_proba > self.threshold
        else:
            return Y_pred_proba == Y_pred_proba.max(axis=1)[:, None]

    def predict_proba(self, X):
        """
        Predict probabilities for classes on text data X

        Args:
           X: list or numpy arrray of texts (n_samples)
        Returns:
           Y_pred_proba: numpy array of probabilities for each class (n_samples, n_classes)
        """
        return self._predict_proba(X)

    def score(self, X, Y):
        """Micro f1 score of X data against ground truth Y"""
        Y_pred = self.predict(X)
        return f1_score(Y, Y_pred, average="micro")

    def save(self, model_path):
        """Saves model in directory model_path"""
        os.makedirs(model_path, exist_ok=True)
        self.model.save_pretrained(model_path)

    def load(self, model_path):
        """Loads model from directory model_path. Tokenizer initialised from param pretrained"""
        self._init_model()
        self.model = TFBertForSequenceClassification.from_pretrained(model_path)
