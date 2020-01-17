.DEFAULT_GOAL := all

VIRTUALENV := build/virtualenv
PYTHON_VERSION := python3.7

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
	echo "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz#egg=en_core_web_sm" >> requirements.txt
	echo "https://github.com/explosion/spacy-models/releases/download/en_trf_bertbaseuncased_lg-2.2.0/en_trf_bertbaseuncased_lg-2.2.0.tar.gz#egg=en_trf_bertbaseuncased_lg" >> requirements.txt

.PHONY: dist
dist:
	$(VIRTUALENV)/bin/python3 setup.py sdist bdist_wheel
	aws s3 sync dist/ s3://datalabs-packages/wellcomeml
	aws s3 cp --recursive --acl public-read dist/ s3://datalabs-public/wellcomeml

# Spacy is require for testing spacy_to_prodigy

$(VIRTUALENV)/.en_core_web_sm: 
	$(VIRTUALENV)/bin/python -m spacy download en_core_web_sm
	touch $@

$(VIRTUALENV)/.en_trf_bertbaseuncased_lg:
	$(VIRTUALENV)/bin/python -m spacy download en_trf_bertbaseuncased_lg
	touch $@

.PHONY: test
test: $(VIRTUALENV)/.en_core_web_sm $(VIRTUALENV)/.en_trf_bertbaseuncased_lg
	$(VIRTUALENV)/bin/pytest -m "not integration" --disable-warnings --tb=line ./tests

.PHONY: test-integrations
test-integrations:
	$(VIRTUALENV)/bin/pytest -m "integration" --disable-warnings --tb=line ./tests

all: virtualenv
