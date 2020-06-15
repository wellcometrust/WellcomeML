.DEFAULT_GOAL := all

VIRTUALENV := build/virtualenv
PYTHON_VERSION := python3

$(VIRTUALENV)/.installed:
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python $(PYTHON_VERSION) $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip3 install -r requirements.txt
	$(VIRTUALENV)/bin/pip3 install -r requirements_test.txt
	$(VIRTUALENV)/bin/pip3 install -e .
	$(VIRTUALENV)/bin/pip3 install git+https://github.com/epfml/sent2vec.git # otherwise it fails cause setup.py relies on cython and numpy being installed
	touch $@

.PHONY: virtualenv
virtualenv: $(VIRTUALENV)/.installed

#
# Tooling for updating requirements.txt, b/c $(VIRTUALENV) also has test
# dependencies in it.
#

.PHONY: update-requirements-txt
update-requirements-txt: VIRTUALENV := build/tmp/update-requirements-virtualenv
update-requirements-txt:
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python $(PYTHON_VERSION) $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip3 install -r unpinned_requirements.txt
	echo "# Created by 'make update-requirements-txt'. DO NOT EDIT!" > requirements.txt
	$(VIRTUALENV)/bin/pip freeze | grep -v pkg-resources==0.0.0 >> requirements.txt

.PHONY: dist
dist:
	./create_release.sh

# Spacy is require for testing spacy_to_prodigy

$(VIRTUALENV)/.models:
	$(VIRTUALENV)/bin/python -m spacy download en_core_web_sm
	touch $@

$(VIRTUALENV)/.deep_learning_models:
	$(VIRTUALENV)/bin/python -m spacy download en_trf_bertbaseuncased_lg
	touch $@

$(VIRTUALENV)/.non_pypi_packages:
	$(VIRTUALENV)/bin/pip install git+https://github.com/epfml/sent2vec.git
	touch $@

.PHONY: download_models
download_models: $(VIRTUALENV)/.installed $(VIRTUALENV)/.models

.PHONY: download_deep_learning_models
download_deep_learning_models: $(VIRTUALENV)/.models $(VIRTUALENV)/.deep_learning_models

.PHONY: download_nonpypi_packages
download_nonpypi_packages: $(VIRTUALENV)/.installed $(VIRTUALENV)/.non_pypi_packages

.PHONY: test
test: $(VIRTUALENV)/.models $(VIRTUALENV)/.deep_learning_models
	$(VIRTUALENV)/bin/pytest -m "not (integration or transformers)" --disable-warnings --tb=line --cov=wellcomeml ./tests

.PHONY: test-transformers
test-transformers:
	$(VIRTUALENV)/bin/pip install -r requirements_transformers.txt
	export WELLCOMEML_ENV=development_transformers && $(VIRTUALENV)/bin/pytest -m "transformers" --disable-warnings --cov-append --tb=line --cov=wellcomeml ./tests/transformers


.PHONY: test-integrations
test-integrations:
	$(VIRTUALENV)/bin/pytest -m "integration" --disable-warnings --tb=line ./tests

all: virtualenv
