import os
import warnings

DISABLE_DIRECT_IMPORTS = os.getenv('DISABLE_DIRECT_IMPORTS', 0)

if not DISABLE_DIRECT_IMPORTS:
    warnings.warn("In the future, importing classes directly from `wellcomeml.ml` will be "
                  "deprecated. Use the full path instead"
                  " (e.g. `from wellcomem.ml.bert_classifier import BertClassifier')",
                  FutureWarning)
    # Modules that need spacy

    from .frequency_vectorizer import WellcomeTfidf
    from .spacy_knowledge_base import SpacyKnowledgeBase
    from .spacy_entity_linking import SpacyEntityLinker
    from .spacy_ner import SpacyNER
    from .spacy_classifier import SpacyClassifier

    # Modules that need tensorflow

    from .bert_classifier import BertClassifier
    from .vectorizer import Vectorizer
    from .cnn import CNNClassifier
    from .bilstm import BiLSTMClassifier
    from .keras_vectorizer import KerasVectorizer
    from .bert_semantic_equivalence import SemanticEquivalenceClassifier
    from .transformers_tokenizer import TransformersTokenizer
    from .clustering import TextClustering

    # Others
    from .bert_vectorizer import BertVectorizer
    from .similarity_entity_linking import SimilarityEntityLinker
    from .doc2vec_vectorizer import Doc2VecVectorizer
    from .sent2vec_vectorizer import Sent2VecVectorizer
    from .voting_classifier import WellcomeVotingClassifier

    __all__ = ['WellcomeTfidf', 'Doc2VecVectorizer',
               'Sent2VecVectorizer', 'WellcomeVotingClassifier',
               'Vectorizer', 'TextClustering', 'SpacyNER', 'SpacyClassifier',
               'BertClassifier', 'BertVectorizer', 'SpacyKnowledgeBase',
               'SpacyEntityLinker', 'SemanticEquivalenceClassifier',
               'CNNClassifier', 'BiLSTMClassifier', 'KerasVectorizer',
               'SimilarityEntityLinker', 'SemanticEquivalenceClassifier',
               'TransformersTokenizer']
