import sys
from unittest import mock

from importlib import import_module

import pytest


extra_checks = {
    'tensorflow': [
        'wellcomeml.ml.attention',
        'wellcomeml.ml.bert_semantic_equivalence',
        'wellcomeml.ml.bilstm',
        'wellcomeml.ml.cnn',
        'wellcomeml.ml.keras_utils',
        #'wellcomeml.ml.keras_vectorizer',
        'wellcomeml.ml.similarity_entity_liking'
    ],
    'torch': [
        #'wellcomeml.ml.bert_vectorizer',
        'wellcomeml.ml.spacy_classifier'
    ],
    'spacy': [
        'wellcomeml.ml.spacy_classifier',
        'wellcomeml.ml.spacy_entity_linking',
        'wellcomeml.ml.spacy_knowledge_base'
    ]
}

module_extra_pairs = [
    (module_name, extra_name)
    for extra_name, module_name_list in extra_checks.items()
    for module_name in module_name_list
]


def extra_import_check(module_name, extra_name):
    with mock.patch.dict(sys.modules, {extra_name: None}):
        import_module(module_name)


@pytest.mark.parametrize("module_name,extra_name", module_extra_pairs)
def test_dependencies(module_name, extra_name):
    """ Tests that importing the module, in the absence of the extra, throws an error """
    with mock.patch.dict(sys.modules, {extra_name: None}):
        with pytest.raises(ImportError):
            extra_import_check(module_name=module_name, extra_name=extra_name)
