[![Build Status](https://travis-ci.com/wellcometrust/WellcomeML.svg?token=cssCZpnz8YDs4Hb4K5pS&branch=master)](https://travis-ci.com/wellcometrust/WellcomeML) [![codecov](https://codecov.io/gh/wellcometrust/wellcomeml/branch/master/graph/badge.svg)](https://codecov.io/gh/wellcometrust/wellcomeml)
![GitHub](https://img.shields.io/github/license/wellcometrust/wellcomeml)
![PyPI](https://img.shields.io/pypi/v/wellcomeml)


# WellcomeML utils

This package contains common utility functions for usual tasks at Wellcome Data Labs, in particular functionalities for processing, embedding and classifying text data. This includes

* An intuitive sklearn-like API wrapping text vectorizers, such as Doc2vec, Bert, Scibert
* Common API for off-the-shelf classifiers to allow quick iteration (e.g. Frequency Vectorizer, Bert, Scibert, basic CNN, BiLSTM)
* Utils to download and convert academic text datasets for benchmark

For more information read the official [![docs](https://img.shields.io/badge/docs-%20-success)](http://wellcometrust.github.io/WellcomeML)


## 1. Quickstart

Installing from PyPi

```bash
pip install wellcomeml
```

This will install the "vanilla" package. In order to install the deep-learning functionality
(torch/transformers/spacy transformers):

```bash
pip install wellcomeml[deep-learning]
```

Installing from a release wheel

Download the wheel [from aws](https://datalabs-public.s3.eu-west-2.amazonaws.com/wellcomeml/wellcomeml-2020.1.0-py3-none-any.whl)
and pip install it:

```bash
pip install wellcomeml-2020.1.0-py3-none-any.whl
pip install wellcomeml-2020.1.0-py3-none-any.whl[deep-learning]
```

### 1.1 Installing wellcomeml[deep-learning] on windows 

Torch has a different installation for windows so it will not get automatically installed with wellcomeml[deeplearning].
It needs to be installed first (this is for machines with no CUDA parallel computing platform for those that do look here https://pytorch.org/ for correct installation):

```
pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

Then install wellcomeml[deep-learning]:

```
pip install wellcomeml[deep-learning]
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

### 2.3 Build the wheel (and upload to aws s3/pypi/github)

Create a [github token](https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line) with artifact write access and export it to the env variables:
```bash
export GITHUB_TOKEN=...
```
After making changes, in order to build a new release, run:

```
make dist
```

### 2.4 (Optional) Installing from other locations

```
pip3 install <relative path to this folder>
```

### 2.5 Transformers

Some experimental features (currently `wellcomeml.ml.SemanticEquivalenceClassifier`) require a version of `transformers` that is not compatible with `spacy-transformers`. To develop those features:

```bash
export WELLCOMEML_ENV=development_transformers
pip install -r requirements_transformers.txt --upgrade
```

On OSX, if you get a message complaining about the rust compiler, install and initialise it with:

```
brew install rustup
rustup-init
```

## 3. Example usage of some modules

Examples can be found in the subfolder `examples`.

