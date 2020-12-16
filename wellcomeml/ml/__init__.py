from wellcomeml.logger import logger


from .frequency_vectorizer import WellcomeTfidf
from .doc2vec_vectorizer import Doc2VecVectorizer
from .sent2vec_vectorizer import Sent2VecVectorizer
from .voting_classifier import WellcomeVotingClassifier
__all__ = ['WellcomeTfidf', 'Doc2VecVectorizer',
           'Sent2VecVectorizer', 'WellcomeVotingClassifier']
try:
    from .vectorizer import Vectorizer
    from .clustering import TextClustering
    from .spacy_ner import SpacyNER
    from .spacy_classifier import SpacyClassifier
    from .bert_classifier import BertClassifier
    from .bert_vectorizer import BertVectorizer
    from .spacy_knowledge_base import SpacyKnowledgeBase
    from .spacy_entity_linking import SpacyEntityLinker
    from .similarity_entity_linking import SimilarityEntityLinker
    from .cnn import CNNClassifier
    from .bilstm import BiLSTMClassifier
    from .keras_vectorizer import KerasVectorizer
    from .bert_semantic_equivalence import SemanticEquivalenceClassifier
    __all__ += [
        'Vectorizer', 'TextClustering', 'SpacyNER', 'SpacyClassifier',
        'BertClassifier', 'BertVectorizer', 'SpacyKnowledgeBase',
        'SpacyEntityLinker', 'SemanticEquivalenceClassifier',
        'CNNClassifier', 'BiLSTMClassifier', 'KerasVectorizer',
        'SimilarityEntityLinker', 'SemanticEquivalenceClassifier'
    ]
except ImportError as e:
    logger.error(e)
    logger.warning("Using WellcomeML without extras (transformers & torch).")
