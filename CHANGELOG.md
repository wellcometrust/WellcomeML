# 2020.07.0

Models
---
* Adds Doc2VecVectorizer
* Adds WellcomeVotingClassifier
* Adds Sent2VecVectorizer
* Adds SemanticEquivalenceMetaClassifier
* Adds CategoricalMetrics and MetricMiniBatchHistory

Datasets
---
* Adds CONLL dataset
* Adds Winer dataset

Features
---
* Automatically load models like en_core_web_sm and en_trf_bertbaseuncased_lg but also download packages like sent2vec, only when needed
* Adds docs based on sphinx and read the docs

Repo
---
* Adds pep8 / flake8 checks and address violations
* Adds badges for build, codecov and license
* Adds pull request template that forces link to issue or trello

Bugs
---
* Fix dependency on non pypi packages for tests
* Pin spacy transformers to 0.5.1
* Fix codecov running in separate travis venv

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
