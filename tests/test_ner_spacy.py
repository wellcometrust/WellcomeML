#!/usr/bin/env python3
# coding: utf-8

import en_core_web_sm
import pytest
from wellcomeml.ml import SpacyNER
from wellcomeml.metrics.ner_classification_report import ner_classification_report


@pytest.fixture(scope="module")
def nlp():
    return en_core_web_sm.load()


@pytest.fixture(scope="module")
def X_train():
    return [
        """n Journal of Psychiatry 158: 2071–4\nFreeman MP,
           Hibbeln JR, Wisner KL et al. (2006)\nOmega-3 fatty ac""",
        """rd, (BKKBN)\n \nJakarta, Indonesia\n29. Drs Titut Prihyugiarto
           \n MSPA\n \nSenior Researcher for Reproducti""",
        """a Santé, 2008. \n118. Konradsen, F. et coll.
           Community uptake of safe storage boxes to reduce self-po""",
        """ted that the two treatments can \nbe combined. Contrarily,
           Wapf et al. \nnoted that many treatment per""",
        """ti-tuberculosis treatment in Mongolia. Int J Tuberc Lung Dis.
           2015;19(6):657–62. \n160. Dudley L, Aze""",
        """he \nScottish Heart Health Study: cohort study. BMJ, 1997, 315:722–729.
           \nUmesawa M, Iso H, Date C et """,
        """T.A., G. Marland, and R.J. Andres (2010). Global, Regional,
           and National Fossil-Fuel CO2 Emissions. """,
        """Ian Gr\nMr Ian Graayy\nPrincipal Policy Officer
           (Public Health and Health Protection), Chartered Insti""",
        """. \n3. \nFischer G and Stöver H. Assessing the current
           state of opioid-dependence treatment across Eur""",
        """ated by\nLlorca et al. (2014) or Pae et al. (2015), or when
           vortioxetine was assumed to be\nas effecti""",
    ]


@pytest.fixture(scope="module")
def y_train():
    return [
        [{'start': 36, 'end': 46, 'label': 'PERSON'},
         {'start': 48, 'end': 58, 'label': 'PERSON'},
         {'start': 61, 'end': 69, 'label': 'PERSON'}],
        [{'start': 41, 'end': 59, 'label': 'PERSON'}],
        [{'start': 21, 'end': 34, 'label': 'PERSON'}],
        [{'start': 58, 'end': 62, 'label': 'PERSON'}],
        [{'start': 87, 'end': 95, 'label': 'PERSON'}],
        [{'start': 72, 'end': 81, 'label': 'PERSON'},
         {'start': 83, 'end': 88, 'label': 'PERSON'},
         {'start': 90, 'end': 96, 'label': 'PERSON'}],
        [{'start': 6, 'end': 16, 'label': 'PERSON'}, {'start': 22, 'end': 33, 'label': 'PERSON'}],
        [{'start': 0, 'end': 6, 'label': 'PERSON'}, {'start': 10, 'end': 20, 'label': 'PERSON'}],
        [{'start': 7, 'end': 16, 'label': 'PERSON'}, {'start': 21, 'end': 30, 'label': 'PERSON'}],
        [{'start': 8, 'end': 14, 'label': 'PERSON'}, {'start': 32, 'end': 35, 'label': 'PERSON'}],
    ]


@pytest.fixture(scope="module")
def ner_groups():
    return ['Group 1', 'Group 2', 'Group 3', 'Group 2', 'Group 1', 'Group 3', 'Group 3',
            'Group 3', 'Group 2', 'Group 1']


def test_fit(X_train, y_train):
    spacy_ner = SpacyNER(n_iter=3, dropout=0.2, output=True)
    spacy_ner.load("en_core_web_sm")
    retrained_nlp = spacy_ner.fit(X_train, y_train)

    assert vars(retrained_nlp)['_meta']['name'] == 'core_web_sm'


def test_predict():
    # Using spaCy's nlp model (don't retrain)
    spacy_ner = SpacyNER(n_iter=3, dropout=0.2, output=True)
    spacy_ner.load("en_core_web_sm")
    pred_entities = spacy_ner.predict("Apple is looking at buying U.K. startup for $1 billion")

    # Make sure its not an empty list
    assert pred_entities


def test_score(X_train, y_train):
    # Using spaCy's nlp model (don't retrain)
    spacy_ner = SpacyNER(n_iter=3, dropout=0.2, output=True)
    spacy_ner.load("en_core_web_sm")
    y_pred = [spacy_ner.predict(text) for text in X_train]
    f1 = spacy_ner.score(y_train, y_pred, tags=['PERSON'])

    assert isinstance(f1['PERSON'], float)


def test_ner_classification_report(X_train, y_train, ner_groups):
    spacy_ner = SpacyNER(n_iter=3, dropout=0.2, output=True)
    spacy_ner.load("en_core_web_sm")
    y_pred = [spacy_ner.predict(text) for text in X_train]
    report = ner_classification_report(y_train, y_pred, ner_groups, tags=['PERSON'])

    assert isinstance(report, str)
