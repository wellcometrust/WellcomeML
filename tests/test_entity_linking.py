import pytest
from wellcomeml.ml.similarity_entity_linking import SimilarityEntityLinker


@pytest.fixture(scope="module")
def entities_kb():
    return {
        'id_1': "American actress. She is the recipient of several accolades, including two Golden"
                " Globe Awards and a Primetime Emmy Award, in addition to nominations for four"
                " Academy Awards and one Tony Award.",
        'id_2': "American entertainer. She rose to fame in the 2000s as a member of R&B girl group"
                " Destiny's Child, one of the best-selling female groups of all time with over"
                " 60 million records, of which more than 35 million copies sold with the trio"
                " lineup with Williams.",
        'id_3': "  "
        }


@pytest.fixture(scope="module")
def stopwords():
    return ['the', 'and', 'if', 'in', 'a']


@pytest.fixture(scope="module")
def train_data():
    return [
        ("After Destiny's Child's disbanded in 2006, Michelle Williams released her first "
         "pop album, Unexpected (2008),", {'id': 'id_2'}),
        ("On Broadway, Michelle Williams starred in revivals of the musical Cabaret in 2014"
         " and the drama Blackbird in 2016, for which she received a nomination for the Tony Award"
         " for Best Actress in a Play.", {'id': 'id_1'}),
        ("Franklin would have ideally been awarded a Nobel Prize in Chemistry", {'id': 'No ID'})
        ]


def test_clean_kb(entities_kb, stopwords):

    entity_linker = SimilarityEntityLinker(stopwords=stopwords)
    knowledge_base = entity_linker._clean_kb(entities_kb)

    assert len(knowledge_base) == 2


def test_optimise_threshold(entities_kb, stopwords, train_data):
    entity_linker = SimilarityEntityLinker(stopwords=stopwords)
    entity_linker.fit(entities_kb)
    entity_linker.optimise_threshold(train_data, id_col='id', no_id_col='No ID')
    optimal_threshold = entity_linker.optimal_threshold

    assert isinstance(optimal_threshold, float)


def test_predict_lowthreshold(entities_kb, stopwords, train_data):
    entity_linker = SimilarityEntityLinker(stopwords=stopwords)
    entity_linker.fit(entities_kb)
    predictions = entity_linker.predict(train_data, similarity_threshold=0.1, no_id_col='No ID')

    assert predictions == ['id_2', 'id_1', 'No ID']


def test_predict_highthreshold(entities_kb, stopwords, train_data):
    entity_linker = SimilarityEntityLinker(stopwords=stopwords)
    entity_linker.fit(entities_kb)
    predictions = entity_linker.predict(train_data, similarity_threshold=1.0, no_id_col='No ID')

    assert predictions == ['No ID', 'No ID', 'No ID']
