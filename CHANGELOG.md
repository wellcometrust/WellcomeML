# 2.0.1
Release date; 24/11/2021

- #376: Fixes a bug with relative import of EPMC API client

# 2.0.0
Release date: 05/11/2021

Improvements:
- #328: Expose build_model to CNNClassifier and don't rebuild in fit
- #372: Add more fine grained extras to make vanilla WellcomeML light (109MB)
- #368: Expand LOGGING_LEVEL to TF_CPP_MIN_LOG_LEVEL
- #332: Add `wellcomeml.viz` to vizualize clusters
- #344: Add filter by variable in visualize clusters

Breaking changes:
- #371: Delete `wellcomeml.ml.__init__` so all ml models need to be explicitly imported

Bug fixes:
- #357: Fix sent2vec test error


# 1.2.1
Release date: 05/08/2021

Bug fixes:

- #346: Fixes problem with tf-idf vectoriser lemmatising twice
- #337: Pin spacy to fix problem with enity-linking test

# 1.2.0
Release date: 22/07/2021

Improvements:
- #327: Adds verbose and tensorboard_log_path to CNN and SemanticEquivalenceClassifier
- #315: Implements decode to KerasTokenizer and TransformersTokenizer.
- #308: Adds EPMCClient to download data from EPMC
- #283: Disable umap to make imports in wellcomeml faster
- #279: Extend LOGGING_LEVEL env variable to control more libraries logger
- #306: Break down deep-learning extra to tensorflow,torch,spacy for more control over what's installed
- #289: Re-factors frequency vectoriser saving function to make it more efficient

Bug fixes:
- #313: Fix concat feature_approach in CNN
- #297: Fix OOM error in CNN predict
- #292: Fix clustering pipeline when umap used and input length > 4096

# 1.1.0
Release date: 26/04/2021

Improvements

- #140: Additions to text clustering, exposing each step of the pipeline and adding load/save
- #272: Spacy lemmatiser speed greatly improved
- #255: Improve memory efficiency of TransformersTokenizer
- #245: Voting classifier extras (better input flexibility, parameter for how many models should agree, etc)

- General improvements to continuous testing pipelines

Bug fixes:

- #233: Fix CNN's dataset generator
- #240: Fix multiGPU in SemanticEquivalenceBer
- #237: Fix KerasVectorizer return length

# 1.0.2
Release date: 25/02/2021

Due to an error, some of the improvements in the previous release weren't built in the PyPI whl.

- #233 CNN predict when X tf.data.Dataset 
- #214 Configurable logs
- #204 Efficient CNN enabling tensorflow datasets as input
- #193 TransfomersTokenizer class

Bug fixes:
- Disables HDBScan for clustering, to solve #197.

# 1.0.1

Release date: 24/02/2021

From now on we will be using Semantic Versioning, instead of the previous CalVer system. Previous versions will still be installabe by pinning it via pip (e.g. `pip install wellcomeml==2021.2.0`), but won't be listed in the PyPI registry anymore. See discussion [here](https://github.com/wellcometrust/WellcomeML/issues/202) for more details.

Improvements
- #211 Batch prediction for `wellcomeml.ml.SemanticSimilarityClassifier`
- ~~#214 Configurable logs~~ only avaliable in 1.0.2
- ~~#204 Efficient CNN enabling tensorflow datasets as input~~ only available in 1.0.2
- ~~#193 TransfomersTokenizer class~~ only available in 1.0.2

Bug fixes:
- ~~Disables HDBScan for clustering, to solve #197.~~

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
