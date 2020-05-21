# 2020.5.1

## ML
* Add l2 and validation_split to BertClassifier
* Add kwargs and load model to semantic similarity

## Extras
* Add command line interface to download models
* Removed models from deployment
* Moved package to pypi


# 2020.5.0

## ML

* Add CNNClassifier
* Add BiLSTMClassifier
* Add attention layers
* Add Semantic equivalence classifier
* Add embedding based entity linker

## Datasets

* Add Hoc dataset

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
