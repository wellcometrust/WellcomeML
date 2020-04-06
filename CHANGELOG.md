# 2020.4.0

* Add partial_fit to BERTClassifier
* Add mean_last_four embedding to BertVectorizer
* Use nlp.pipe for prediction as its quicker
* Add generator to transform data on demand for spacy to reduce memory usage
* Add multilabel and architecture parameter in SpacyClassifier
* Modify SpacyClassifier to accept sparse Y for multilabel classification
* Add pretrain_vectors_path parameters to SpacyClassifier
* Add speed metric to SpacyClassifier and BertClassifier
* Fix tests in BertClassifier to check for loss reduction after 5 iterations
