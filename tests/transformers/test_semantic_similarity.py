# encoding: utf-8
import pytest

from wellcomeml.ml.bert_semantic_equivalence import SemanticEquivalenceClassifier


@pytest.mark.transformers
def test_semantic_similarity():
    classifier = SemanticEquivalenceClassifier(pretrained="scibert",
                                               batch_size=2,
                                               eval_batch_size=1)

    X = [('This sentence has context_1', 'This one also has context_1'),
         ('This sentence has context_2', 'This one also has context_2'),
         ('This sentence is about something else', 'God save the queen')]

    y = [1, 1, 0]

    classifier.fit(X, y, epochs=3)

    loss_initial = classifier.history['loss'][0]
    loss_epoch_2 = classifier.history['loss'][2]
    scores = classifier.score(X)

    assert loss_epoch_2 < loss_initial
    assert len(classifier.predict(X)) == 3
    assert (scores > 0).sum() == 6
    assert (scores < 1).sum() == 6

    # Fits two extra epoch

    classifier.fit(X, y, epochs=2)

    # Asserts that the classifier model is adding to the history, and still
    # not re-training from scratch

    assert len(classifier.history['loss']) == 5
