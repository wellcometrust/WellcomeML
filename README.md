[![Build Status](https://travis-ci.com/wellcometrust/WellcomeML.svg?token=cssCZpnz8YDs4Hb4K5pS&branch=master)](https://travis-ci.com/wellcometrust/WellcomeML)

# WellcomeML utils

This package contains common utility functions for usual tasks at Wellcome Data Labs. In particular:


| modules | description| 
|---|---|
| **io** | manipulating data, in and out S3, and processing |
| **ml** | wrappers for processing texts, vectorisers and classifiers |
| **spacy** | common utils for converting data form and to spacy/prodigy format |
| **mis/viz** | any other utils, including Wellcome colour palletes | 



## 1. Getting started

### 1.1 Build local virtualenv

```
make
```

### 1.2 Build the wheel (and upload to aws s3)

After making changes, in order to buil a new wheel, run:

```
make dist
```

### 1.3 (Optional) Installing from other locations within the datalabs monorepo

```
pip3 install <relative path to this folder>
```

## 2. Example usage of some modules

Examples can be found in the subfolder `examples`.

