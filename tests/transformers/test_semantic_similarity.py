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


def test_save_and_load(tmp_path):
    classifier_1 = SemanticEquivalenceClassifier(pretrained="scibert",
                                                 batch_size=2,
                                                 eval_batch_size=1)

    X = [('This sentence has context_1', 'This one also has context_1'),
         ('This sentence has context_2', 'This one also has context_2'),
         ('This sentence is about something else', 'God save the queen')]

    y = [1, 1, 0]

    classifier_1.fit(X, y, epochs=3)
    classifier_1.save(tmp_path)
    scores_1 = classifier_1.score(X)

    classifier_2 = SemanticEquivalenceClassifier(pretrained="scibert")
    classifier_2.load(tmp_path)
    scores_2 = classifier_2.score(X)

    score_diff = sum([abs(diff) for diff in (scores_1-scores_2).flatten()])

    assert pytest.approx(score_diff, 0)
