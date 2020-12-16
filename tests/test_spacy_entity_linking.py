import tempfile
import pytest

from wellcomeml.ml import SpacyKnowledgeBase
from wellcomeml.ml import SpacyEntityLinker


@pytest.fixture(scope="module")
def entities():
    # A dict of each entity, it's description and it's corpus frequency
    return {
        "id_1": (
            "American actress. She is the recipient of several accolades, including two Golden"
            " Globe Awards and a Primetime Emmy Award, in addition to nominations for four"
            " Academy Awards and one Tony Award.",
            0.1,
        ),
        "id_2": (
            "American entertainer. She rose to fame in the 2000s as a member of R&B girl group"
            " Destiny's Child, one of the best-selling female groups of all time with over"
            " 60 million records, of which more than 35 million copies sold with the trio"
            " lineup with Williams.",
            0.05,
        ),
    }


@pytest.fixture(scope="module")
def list_aliases():
    # A list of dicts for each entity
    # probabilities are 'prior probabilities' and must sum to < 1
    return [
        {
            "alias": "Michelle Williams",
            "entities": ["id_1", "id_2"],
            "probabilities": [0.7, 0.3],
        }
    ]


@pytest.fixture(scope="module")
def data():
    return [
        (
            "After Destiny's Child's disbanded in 2006. Michelle Williams released her first "
            "pop album, Unexpected (2008),",
            {"links": {(43, 60): {"id_1": 0.0, "id_2": 1.0}}},
        ),
        (
            "On Broadway, Michelle Williams starred in revivals of the musical Cabaret in 2014"
            " and the drama Blackbird in 2016."
            " For which she received a nomination for the Tony Award"
            " for Best Actress in a Play.",
            {"links": {(13, 30): {"id_1": 1.0, "id_2": 0.0}}},
        ),
    ]


def test_kb_train(entities, list_aliases):

    kb = SpacyKnowledgeBase(kb_model="en_core_web_sm")
    kb.train(entities, list_aliases)

    assert sorted(kb.kb.get_entity_strings()) == ["id_1", "id_2"]
    assert kb.kb.get_alias_strings() == ["Michelle Williams"]


def test_el_train(entities, list_aliases, data):

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_kb = SpacyKnowledgeBase(kb_model="en_core_web_sm")
        temp_kb.train(entities, list_aliases)
        temp_kb.save(tmp_dir)
        el = SpacyEntityLinker(tmp_dir, print_output=False)
        el.train(data)

        assert "entity_linker" in el.nlp.pipe_names


def test_el_predict(entities, list_aliases, data):

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_kb = SpacyKnowledgeBase(kb_model="en_core_web_sm")
        temp_kb.train(entities, list_aliases)
        temp_kb.save(tmp_dir)
        el = SpacyEntityLinker(tmp_dir, print_output=False)
        el.train(data)
        predicted_ids = el.predict(data)

        assert predicted_ids == [["id_2"], ["id_1"]]
