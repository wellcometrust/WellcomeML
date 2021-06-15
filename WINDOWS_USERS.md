# Installation on Windows machines

## Requirements
This package requires an up to date Windows 10, as well as the following:
 - Visual Studio Build tools 2019 with `Desktop Development with C++` installed
 - Python 3.8 installed at the root of your machine (the Makefile will look for it in C://Python38)
 - Administration rights
 - GNU Make
 - The Makefile will assume the `OS` environment variable is set to its default value
 - (Optional but recommended) Cygwin


## Installation
Run the following Makefile:
`make virtualenv`

## Tests
Running tests might take a bit of time on the first run, as you will need to download some models and build a few libraries.
`make test`
