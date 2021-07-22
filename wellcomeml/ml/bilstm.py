from wellcomeml.ml.base_nn import BaseNNClassifier


class BiLSTMClassifier(BaseNNClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, feature_encoder="cnn")
