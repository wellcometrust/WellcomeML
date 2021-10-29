[![Build Status](https://travis-ci.com/wellcometrust/WellcomeML.svg?token=cssCZpnz8YDs4Hb4K5pS&branch=main)](https://travis-ci.com/wellcometrust/WellcomeML) [![codecov](https://codecov.io/gh/wellcometrust/wellcomeml/branch/main/graph/badge.svg)](https://codecov.io/gh/wellcometrust/wellcomeml)
![GitHub](https://img.shields.io/github/license/wellcometrust/wellcomeml)
![PyPI](https://img.shields.io/pypi/v/wellcomeml)
[![docs](https://img.shields.io/badge/docs-%20-success)](http://wellcometrust.github.io/WellcomeML)


# WellcomeML utils

This package contains common utility functions for usual tasks at the Wellcome Trust, in particular functionalities for processing, embedding and classifying text data. This includes

* An intuitive sklearn-like API wrapping text vectorizers, such as Doc2vec, Bert, Scibert
* Common API for off-the-shelf classifiers to allow quick iteration (e.g. Frequency Vectorizer, Bert, Scibert, basic CNN, BiLSTM, SemanticSimilarity)
* Utils to download and convert academic text datasets for benchmark
* Utils to download data from the EPMC API

For more information read the official [docs](http://wellcometrust.github.io/WellcomeML).


## 1. Quickstart

Installing from PyPi

```bash
pip install wellcomeml
```

This will install the "vanilla" package with very little functionality, such as io, dataset download etc.

If space is not a problem, you can install the  _full_ package (around 2.2GB):

```bash
pip install wellcomeml[all]
```

The full package is relatively big, therefore we also have fine-grained installations if you only wish to use one specific module.
Those are `core, transformers, tensorflow, torch, spacy`. You can install one, or more of those you want, e.g.:

```bash
pip install wellcomeml[tensorflow, core]
```

To check that your installation allows you to use a specific module, try (for example):

```bash
python -c "import wellcomeml.ml.bert_vectorizer"
```

If you don't have the correct dependencies installed for a module, an error will appear 
and point you to the right dependencies.

### 1.1 Installing wellcomeml[all] on windows 

Torch has a different installation for windows so it will not get automatically installed with wellcomeml[all].
It needs to be installed first (this is for machines with no CUDA parallel computing platform for those that do look here https://pytorch.org/ for correct installation):

```
pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install wellcomeml[all]
```

## 2. Development

### 2.1 Build local virtualenv

```
make
```

### 2.2 Contributing to the docs

Make changes to the .rst files in `/docs` (please **do not change the ones starting by wellcomeml** as those are generated automatically)

Navigate to the root repository and run

```bash
make update-docs
```

Verify that `_build/html/index.html` has generated correctly and submit a PR.

### 2.3 Release a new version (and upload to aws s3/pypi/github)

First create a [github token](https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line), if you haven't one, with artifact write access and
 export
 it to the env variables:
```bash
export GITHUB_TOKEN=...
```

The checklist for a new release is:

- [ ] Change `wellcomeml/__version__.py`
- [ ] Add changelog
- [ ] `make dist`
- [ ] Verify new package was generated correctly on the [pip registry](https://pypi.org/project/wellcomeml)
 and GitHub releases 


### 2.4 (Optional) Installing from other locations

```
pip3 install <relative path to this folder>
```

### 2.5 Transformers

On OSX, if you get a message complaining about the rust compiler, install and initialise it with:

```
brew install rustup
rustup-init
```

## 3. Example usage of some modules

Examples can be found in the subfolder `examples`.

## 4. Troubleshooting

If you experience a problem with installing or using WellcomeML please open an issue. It might be
worth setting the logging level to DEBUG `export LOGGING_LEVEL=DEBUG` which will often expose
more information that might be informative to resolve the issue.

## 5. Extras


|Module|Description|Extras needed|
|---|---|---|
| wellcomeml.ml.attention | Classes that implement keras layers for attention/self-attention  |  tensorflow |
| wellcomeml.ml.bert_classifier  | Classifier to facilitate fine-tuning bert/scibert  |  tensorflow |
| wellcomeml.ml.bert_semantic_equivalence  | Classifier to learn semantic equivalence between pairs of documents  |  tensorflow |
| wellcomeml.ml.bert_vectorizer | Text vectorizer based on bert/scibert | torch |
| wellcomeml.ml.bilstm | BILSTM Text classifier | tensorflow |
| wellcomeml.ml.clustering | Text clustering pipeline | NA |
| wellcomeml.ml.cnn | CNN Text Classifier | tensorflow |
| wellcomeml.ml.doc2vec_vectorizer | Text vectorizer based on doc2vec | NA |
| wellcomeml.ml.frequency_vectorizer | Text vectorizer based on TF-IDF | NA |
| wellcomeml.ml.keras_utils | Utils for computing metrics during training | tensorflow |
| wellcomeml.ml.keras_vectorizer | Text vectorizer based on Keras | tensorflow |
| wellcomeml.ml.sent2vec_vectorizer | Text vectorizer based on Sent2Vec | (Requires sent2vec, a non-pypi package) |
| wellcomeml.ml.similarity_entity_liking | A class to find most similar documents to a sentence in a corpus | tensorflow |
| wellcomeml.ml.spacy_classifier | A text classifier based on spacy | spacy, torch |
| wellcomeml.ml.spacy_entity_linking | Similar to similarity_entity_linking, but uses spacy | spacy |
| wellcomeml.ml.spacy_knowledge_base | Creates a knowledge base of entities, based on [spacy](https://spacy.io/usage/training#entity-linker) | spacy |
| wellcomeml.ml.spacy_ner | Named entity recognition classifier based on spacy | spacy |
| wellcomeml.ml.transformers_tokenizer | Bespoke tokenizer based on transformers | Transformers |
| wellcomeml.ml.vectorizer | Abstract class for vectorizers | NA |
| wellcomeml.ml.voting_classifier | Meta-classifier based on majority voting | NA |
