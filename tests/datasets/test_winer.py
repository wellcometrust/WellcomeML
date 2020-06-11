import pytest
import os

from wellcomeml.datasets.winer import create_train_test, _load_data_spacy

@pytest.fixture(scope="module")
def define_paths():

    NE_path = 'tests/test_data/mock_winer_CoarseNE.tar.bz2'
    docs_path ='tests/test_data/mock_winer_Documents.tar.bz2'
    vocab_path = 'tests/test_data/mock_winer_document.vocab'

    # These will be generated and then deleted in these tests
    train_processed_path = 'tests/test_data/temp_train_sample.txt'
    test_processed_path = 'tests/test_data/temp_test_sample.txt'

    return (
        NE_path, docs_path, vocab_path,
        train_processed_path, test_processed_path
        )

def test_create_train_test(define_paths):

    (NE_path, docs_path, vocab_path,
        train_processed_path, test_processed_path) = define_paths
    # Create the train/test data
    n_sample = 10
    prop_train = 0.5

    create_train_test(
        NE_path, vocab_path, docs_path,
        train_processed_path, test_processed_path,
        n_sample, prop_train, rand_seed=42
        )
    assert os.path.exists(train_processed_path) and os.path.exists(test_processed_path)
    os.remove(train_processed_path)
    os.remove(test_processed_path)


def test_train_test_documents(define_paths):

    (NE_path, docs_path, vocab_path,
        train_processed_path, test_processed_path) = define_paths
    # Create the train/test data
    n_sample = 10
    prop_train = 0.5
    # There are 4 documents with entities in the sample data
    expected_train_size = round(prop_train*4) 

    create_train_test(
        NE_path, vocab_path, docs_path,
        train_processed_path, test_processed_path,
        n_sample, prop_train, rand_seed=42
        )

    docs_IDs = ['ID 1000', 'ID 10002', 'ID 12083', 'ID 12084']

    with open(train_processed_path, 'r') as file:
        train_text = file.read()
    with open(test_processed_path, 'r') as file:
        test_text = file.read()

    train_ids = [d for d in docs_IDs if d in train_text]
    test_ids = [d for d in docs_IDs if d in test_text]

    assert len(train_ids) == expected_train_size and \
        len(test_ids) == (4-expected_train_size) and \
        len(set(train_ids+test_ids)) == 4
    os.remove(train_processed_path)
    os.remove(test_processed_path)


def test_length():
    X, Y = _load_data_spacy('tests/test_data/test_winer.txt', inc_outside=True)

    assert len(X) == len(Y) and len(X) == 168

def test_entity():
    X, Y = _load_data_spacy('tests/test_data/test_winer.txt', inc_outside=False)

    start = Y[0][0]['start']
    end = Y[0][0]['end']

    assert X[0][start:end] == 'Communications in Gibraltar'

def test_no_outside_entities():
    X, Y = _load_data_spacy('tests/test_data/test_winer.txt', inc_outside=False)

    outside_entities = [entity for entities in Y for entity in entities if entity['label']=='O']

    assert len(outside_entities) == 0

