from wellcomeml.logger import logger

from .entity_linking import EntityLinker
from .knowledge_base import PeopleKB
from .frequency_vectorizer import WellcomeTfidf

try:
    from .bert_classifier import BertClassifier
    from .bert_vectorizer import BertVectorizer
    from .vectorizer import Vectorizer
    from .spacy_ner import SpacyNER
    from .spacy_classifier import SpacyClassifier
    from .cnn import CNNClassifier
    from .bilstm import BiLSTMClassifier
    from .keras_vectorizer import KerasVectorizer
except ImportError:
    logger.warning("Using WellcomeML without extras (transformers & torch).")
