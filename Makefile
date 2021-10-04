.DEFAULT_GOAL := all

VIRTUALENV := build/virtualenv

ifeq ($(OS), Windows_NT)
	# for CYGWIN*|MINGW32*|MSYS*|MINGW*
	PYTHON_VERSION := C://Python38/python
	VENV_BIN := $(VIRTUALENV)/Scripts
else
	PYTHON_VERSION := python3.8
	VENV_BIN := $(VIRTUALENV)/bin
endif

$(VIRTUALENV)/.installed:
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	$(PYTHON_VERSION) -m venv $(VIRTUALENV)
	$(VENV_BIN)/pip3 install --upgrade pip
	$(VENV_BIN)/pip3 install -r requirements_test.txt
	$(VENV_BIN)/pip3 install -r docs/requirements.txt # Installs requirements to docs
	$(VENV_BIN)/pip3 install -e .[tensorflow,spacy,torch]
	$(VENV_BIN)/pip3 install hdbscan --no-cache-dir --no-binary :all: --no-build-isolation
	touch $@

.PHONY: update-docs
update-docs:
	$(VENV_BIN)/sphinx-apidoc --no-toc -d 5 -H WellcomeML -o ./docs -f wellcomeml
	. $(VENV_BIN)/activate && cd docs && make html

.PHONY: virtualenv
virtualenv: $(VIRTUALENV)/.installed

.PHONY: dist
dist: update-docs
	./create_release.sh

# Spacy is require for testing spacy_to_prodigy

$(VIRTUALENV)/.models:
	$(VENV_BIN)/python -m spacy download en_core_web_sm
	touch $@

$(VIRTUALENV)/.deep_learning_models:
#	$(VENV_BIN)/python -m spacy download en_trf_bertbaseuncased_lg
	touch $@

$(VIRTUALENV)/.non_pypi_packages:
	# Install from local git directory - pip install [address] fails on Windows
	git clone https://github.com/epfml/sent2vec.git
	cd sent2vec && git checkout f00a1b67f4330e5be99e7cc31ac28df94deed9ac && $(VENV_BIN)/pip install . # Install latest compatible sent2vec
	@rm -rf sent2vec
	touch $@

.PHONY: download_models
download_models: $(VIRTUALENV)/.installed $(VIRTUALENV)/.models

.PHONY: download_deep_learning_models
download_deep_learning_models: $(VIRTUALENV)/.models $(VIRTUALENV)/.deep_learning_models

.PHONY: download_nonpypi_packages
download_nonpypi_packages: $(VIRTUALENV)/.installed $(VIRTUALENV)/.non_pypi_packages

.PHONY: test
test: $(VIRTUALENV)/.models $(VIRTUALENV)/.deep_learning_models $(VIRTUALENV)/.non_pypi_packages
	$(VENV_BIN)/tox

.PHONY: test-integrations
test-integrations:
	$(VENV_BIN)/pytest -m "integration" -s -v --disable-warnings --tb=line ./tests

.PHONY: run_codecov
run_codecov:
	$(VENV_BIN)/python -m codecov

all: virtualenv test
