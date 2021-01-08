import pytest

from wellcomeml.ml import SpacyKnowledgeBase


@pytest.fixture(scope="module")
def entities():
    # A dict of each entity, it's description and it's corpus frequency
    return {
        'id_1': (
            "American actress. She is the recipient of several accolades, including two Golden"
            " Globe Awards and a Primetime Emmy Award, in addition to nominations for four"
            " Academy Awards and one Tony Award.", 0.1),
        'id_2': (
            "American entertainer. She rose to fame in the 2000s as a member of R&B girl group"
            " Destiny's Child, one of the best-selling female groups of all time with over"
            " 60 million records, of which more than 35 million copies sold with the trio"
            " lineup with Williams.", 0.05)
        }


@pytest.fixture(scope="module")
def list_aliases():
    # A list of dicts for each entity
    # probabilities are 'prior probabilities' and must sum to < 1
    return [{
        'alias': 'Michelle Williams',
        'entities': ['id_1', 'id_2'],
        'probabilities': [0.7, 0.3]
                    }]


def test_kb_train(entities, list_aliases):

    kb = SpacyKnowledgeBase(kb_model="en_core_web_md")
    kb.train(entities, list_aliases)

    assert sorted(kb.kb.get_entity_strings()) == ['id_1', 'id_2']
    assert kb.kb.get_alias_strings() == ['Michelle Williams']
