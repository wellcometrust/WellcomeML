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
	$(VIRTUALENV)/bin/python3 setup.py sdist bdist_wheel
	aws s3 sync dist/ s3://datalabs-packages/wellcomeml
	aws s3 cp --recursive --acl public-read dist/ s3://datalabs-public/wellcomeml
	python -m twine upload --repository testpypi --username $TWINE_USERNAME --password $TWINE_PASSWORD dist/*

# Spacy is require for testing spacy_to_prodigy

$(VIRTUALENV)/.models: 
	$(VIRTUALENV)/bin/python -m wellcomeml download models
	touch $@

$(VIRTUALENV)/.deep_learning_models:
	$(VIRTUALENV)/bin/python -m wellcomeml download deeplearning-models
	touch $@

.PHONY: download_models
download_models: $(VIRTUALENV)/.models

.PHONY: download_deep_learning_models
download_deep_learning_models: $(VIRTUALENV)/.models $(VIRTUALENV)/.deep_learning_models

.PHONY: test
test: $(VIRTUALENV)/.models $(VIRTUALENV)/.deeplearning-models
	$(VIRTUALENV)/bin/pytest -m "not (integration or transformers)" --disable-warnings --tb=line --cov=wellcomeml ./tests

.PHONY: test-transformers
test-transformers:
	$(VIRTUALENV)/bin/pip install -r requirements_transformers.txt
	export WELLCOMEML_ENV=development_transformers && $(VIRTUALENV)/bin/pytest -m "transformers" --disable-warnings --cov-append --tb=line --cov=wellcomeml ./tests/transformers


.PHONY: test-integrations
test-integrations:
	$(VIRTUALENV)/bin/pytest -m "integration" --disable-warnings --tb=line ./tests

all: virtualenv
