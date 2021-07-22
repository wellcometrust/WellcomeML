"""
CNN architecture inspired from spacy for NLP tasks

It follows the embed, encode, attend, predict framework

    Embed: learns a new embedding but can receive
    pre trained embeddings as well

    Encode: stacked CNN with context window 3 that maintains the size of input and applies dropout
    and layer norm

    Attend: not yet implemented

    Predict: softmax or sigmoid depending on number of outputs
    and whether task is multilabel
"""
from wellcomeml.ml.base_nn import BaseNNClassifier


class CNNClassifier(BaseNNClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, feature_encoder="cnn")
