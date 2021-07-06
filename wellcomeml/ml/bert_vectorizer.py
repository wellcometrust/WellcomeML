# encoding: utf-8
"""
BERT Vectorizer that embeds text using a prertained BERT model
"""
import logging

from transformers import BertModel, BertTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import tqdm

from wellcomeml.utils import check_cache_and_download, throw_extra_import_message

try:
    import torch
except ImportError as e:
    throw_extra_import_message(error=e, extra="torch", required_module="torch")

logger = logging.getLogger(__name__)


class BertVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, pretrained="bert", sentence_embedding="mean_second_to_last"):
        """
        Bert vectorizer parameters

        Args:
            pretrained: A pre-trained model name. Currently 'bert' or 'scibert'
            sentence_embedding: How to embedd a sentence using bert's layers.
            Current options:
            'mean_second_to_last', 'mean_last', 'sum_last' or 'mean_last_four'
            Default: `mean_second_to_last`. If a valid option is not set,
            returns the pooler layer (embedding for the token [CLS])
        """
        self.pretrained = pretrained
        self.sentence_embedding = sentence_embedding

    @classmethod
    def save_transformed(cls, path, X_transformed):
        """Saves transformed embedded vectors"""
        np.save(path, X_transformed)

    @classmethod
    def load_transformed(cls, path):
        """Loads transformed embedded vectors"""
        return np.load(path)

    def bert_embedding(self, x):
        tokenized_x = self.tokenizer.tokenize(x)

        # Max sequence length is 512 for BERT. 510 without CLS and SEP
        if len(tokenized_x) > 510:
            embedded_a = self.bert_embedding(" ".join(tokenized_x[:510]))
            embedded_b = self.bert_embedding(" ".join(tokenized_x[510:]))
            return embedded_a + embedded_b

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + tokenized_x + ["[SEP]"])

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.zeros(tokens_tensor.shape, dtype=torch.long)
        with torch.no_grad():
            output = self.model(tokens_tensor, token_type_ids=segments_tensor)
        last_layer = output[2][-1]
        second_to_last_layer = output[2][-2]

        if self.sentence_embedding == "mean_second_to_last":
            embedded_x = second_to_last_layer.mean(dim=1)
        elif self.sentence_embedding == "mean_last":
            embedded_x = last_layer.mean(dim=1)
        elif self.sentence_embedding == "sum_last":
            embedded_x = last_layer.sum(dim=1)
        elif self.sentence_embedding == "mean_last_four":
            embedded_x = torch.stack(output[2][-4:]).mean(dim=0).mean(dim=1)
        else:
            # Else gives the embbedding for the pooler layer. This can be, for
            # example, fed into a softmax classifier
            embedded_x = output[1]

        return embedded_x.cpu().numpy().flatten()

    def transform(self, X, verbose=True, *_):
        X = (tqdm.tqdm(X) if verbose else X)

        return np.array([self.bert_embedding(x) for x in X])

    def fit(self, *_):
        model_name = (
            "bert-base-uncased"
            if self.pretrained == "bert"
            else "scibert_scivocab_uncased"
        )

        # If model_name doesn't exist checks cache and change name to
        # full path
        if model_name == "scibert_scivocab_uncased":
            model_name = check_cache_and_download(model_name)

        logger.info("Using {} embedding".format(model_name))
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model.eval()
        return self
