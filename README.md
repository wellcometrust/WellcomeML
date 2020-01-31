[![Build Status](https://travis-ci.com/wellcometrust/WellcomeML.svg?token=cssCZpnz8YDs4Hb4K5pS&branch=master)](https://travis-ci.com/wellcometrust/WellcomeML)

# WellcomeML utils

This package contains common utility functions for usual tasks at Wellcome Data Labs. In particular:


| modules | description| 
|---|---|
| **io** | manipulating data, in and out S3, and processing |
| **ml** | wrappers for processing texts, vectorisers and classifiers |
| **spacy** | common utils for converting data form and to spacy/prodigy format |
| **mis/viz** | any other utils, including Wellcome colour palletes | 


## 1. Quickstart
Installing from a release wheel: Download the wheel [from aws](https://s3.amazonaws.com/datalabs-public/wellcomeml/wellcomeml-2020.1.0-py3-none-any.whl)
and pip install it:

```bash
pip install wellcomeml-2020.1.0-py3-none-any.whl
```

This will install the "vanilla" package. In order to install the deep-learning functionality
(torch/transformers/spacy transformers):

```bash
pip install wellcomeml-2020.1.0-py3-none-any.whl[deep-learning]
```


## 2. Development

### 2.1 Build local virtualenv

```
make
```

### 2.2 Build the wheel (and upload to aws s3)

After making changes, in order to buil a new wheel, run:

```
make dist
```

### 3.3 (Optional) Installing from other locations

```
pip3 install <relative path to this folder>
```

## 3. Example usage of some modules

Examples can be found in the subfolder `examples`.

