# Installation on Windows mahcines

## Requirements
This package requires an up to date Windows 10, as well as the following:
 - Visual Studio Build tools 2019 with `Desktop Development with C++` installed
 - Python 3.8
 - Administration rights
 - Make
 - (Optional but recommended) Cygwin


## Installation
Run the following Makefile:
`make virtualenv --file Makefile.win`

As a follow-up, every command ran from the makefile should specify `--file Makefile.win`, as there is some differences between the *nix installation and the Windows one.


## Tests
Running tests might take a bit of time on the first run, as you will need to download some models and build a few libraries.
`make test --file Makefile.win`
