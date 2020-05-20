[![Build Status](https://travis-ci.com/wellcometrust/WellcomeML.svg?token=cssCZpnz8YDs4Hb4K5pS&branch=master)](https://travis-ci.com/wellcometrust/WellcomeML) [![codecov](https://codecov.io/gh/wellcometrust/wellcomeml/branch/master/graph/badge.svg)](https://codecov.io/gh/wellcometrust/wellcomeml)

# WellcomeML utils

This package contains common utility functions for usual tasks at Wellcome Data Labs. In particular:


| modules | description| 
|---|---|
| **io** | manipulating data, in and out S3, and processing |
| **ml** | wrappers for processing texts, vectorisers and classifiers |
| **spacy** | common utils for converting data form and to spacy/prodigy format |
| **mis/viz** | any other utils, including Wellcome colour palletes | 

For more in depth information see the `/examples` folder and [release notes](https://github.com/wellcometrust/WellcomeML/releases).

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

## 2. Development

### 2.1 Build local virtualenv

```
make
```

### 2.2 Build the wheel (and upload to aws s3/pypi/github)

Create a [github token](https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line) with artifact write access and export it to the env variables:
```bash
export GITHUB_TOKEN=...
```
After making changes, in order to buil a new wheel, run:

```
make dist
```

### 2.3 (Optional) Installing from other locations

```
pip3 install <relative path to this folder>
```

### 2.4 Transformers

Some experimental features (currently `wellcomeml.ml.SemanticEquivalenceClassifier`) require a version of `transformers` that is not compatible with `spacy-transformers`. To develop those features:

```bash
export WELLCOMEML_ENV=development_transformers
pip install -r requirements_transformers.txt --upgrade
```

On OSX, ff you get a message complaining about the rust compiler, install and initialise it with:

```
brew install rustup
rustup-init
```

## 3. Example usage of some modules

Examples can be found in the subfolder `examples`.

