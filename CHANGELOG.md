# 1.0.1

Release date: 24/02/2021

From now on we will be using Semantic Versioning, instead of the previous CalVer system. Previous versions will still be installabe by pinning it via pip (e.g. `pip install wellcomeml==2021.2.0`), but won't be listed in the PyPI registry anymore. See discussion [here](https://github.com/wellcometrust/WellcomeML/issues/202) for more details.

Improvements
- #211 Batch prediction for `wellcomeml.ml.SemanticSimilarityClassifier`
- #214 Configurable logs
- #204 Efficient CNN enabling tensorflow datasets as input
- #193 TransfomersTokenizer class

Bug fixes:
- Disables HDBScan for clustering, to solve #197.

Notice: for a technical reason, we are starting with 1.0.1 and not 1.0.0

# 2021.2.1

Bug fixes:
- Update version to 2021.2.1 which was out of sync in latest release #206

# 2021.2.0

Major changes
---
- Upgrade spacy to v3.0
- Add native HuggingFace support (#191), re-writting `BertClassifier` using transformers
- Disables HDBscan from the possible clustering techniques due to a conflict with the new numpy version (#197)

Bug fixes
---
- Resolves issues #195 and #198 with thew pip reference resolver, introduced in pip>20.3

# 2021.1.1

Major changes
---
- Upgrade spacy to 2.3, transformers to < 2.9 and spacy-tranformers to 0.6
- Transformers functionality (e.g. `SemanticEquivalenceClassifier` and `SemanticEquivalenceClassifier`) are now enabled automatically with extras 'deep-learning'

Bug fixes
---
- #173 Fix BertVectorizer for long sequences

# 2020.11.1

Bug fixes
---
- #179 Dataclasses dependency version conflict coming from spacy-transformers
- #177 BertClassifier path for scibert when fine tuning not working

# 2020.11.0

Models
---

### CNN/BiLSTM
- predict_proba
- threshold param default 0.5
- learning_rate decay param default 1
- run training on multiple GPUs if more than one GPUs available
- early_stopping param default false
- sparse_y param default false
- save and load methods

### SemanticEquivalenceClassifier
- callbacks param with default tensorboard and minibatch history
- dropout_rate param default 0.1
- dropout param default true
- batch_norm param default true

### KerasVectorizer
- accepts gensim compatible word vector names and downloads and caches them

Metrics
---
- f1 loss and metric

Bug fixes
---
- #157 about saving CNN/BiLSTM with attention layer
- #80  that fixes indentation and other doc issues
- #134 which allows to use tsne as reducer in clustering

# 2020.9.0

Models
---
* Add param l2, dense_size, attention_heads, metrics, callbacks, feature_approach to cnn and bilstm classifiers
* Faster predictions in BertClassifier through the use of spacy's pipe
* Adds TextClustering

# 2020.07.1

Bugs
---
* Fixes pypi conflict by pinning down dependencies.

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
