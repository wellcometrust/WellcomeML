#!/usr/bin/env python3
# coding: utf-8

import os
import sys

import pytest
from wellcomeml.prodigy.prodigy_to_tsv import TokenLabelPairs


def test_yield_token_label_pair():

    doc = dict()

    doc["spans"] = [
        {"start": 0, "end": 0, "token_start": 0, "token_end": 0, "label": "a"},
        {"start": 1, "end": 1, "token_start": 1, "token_end": 1, "label": "b"},
        {"start": 2, "end": 2, "token_start": 2, "token_end": 2, "label": "c"},
        {"start": 3, "end": 3, "token_start": 3, "token_end": 3, "label": "d"},
        {"start": 4, "end": 4, "token_start": 4, "token_end": 4, "label": "e"},
        {"start": 5, "end": 5, "token_start": 5, "token_end": 5, "label": "f"},
        {"start": 6, "end": 6, "token_start": 6, "token_end": 6, "label": "g"},
    ]

    doc["tokens"] = [
        {"text": "A", "start": 0, "end": 0, "id": 0},
        {"text": "B", "start": 1, "end": 1, "id": 1},
        {"text": "C", "start": 2, "end": 2, "id": 2},
        {"text": "D", "start": 3, "end": 3, "id": 3},
        {"text": "E", "start": 4, "end": 4, "id": 4},
        {"text": "F", "start": 5, "end": 5, "id": 5},
        {"text": "G", "start": 6, "end": 6, "id": 6},
    ]

    out = [
        ("A", "a"),
        ("B", "b"),
        ("C", "c"),
        ("D", "d"),
        ("E", "e"),
        ("F", "f"),
        ("G", "g"),
        (None, None),
    ]

    tlp = TokenLabelPairs(line_limit=73, respect_line_endings=True)
    after = list(tlp.yield_token_label_pair(doc))

    assert out == after


def test_TokenLabelPairs():

    doc = dict()

    doc["spans"] = [
        {"start": 0, "end": 0, "token_start": 0, "token_end": 0, "label": "a"},
        {"start": 1, "end": 1, "token_start": 1, "token_end": 1, "label": "b"},
        {"start": 2, "end": 2, "token_start": 2, "token_end": 2, "label": "c"},
        {"start": 3, "end": 3, "token_start": 3, "token_end": 3, "label": "d"},
        {"start": 4, "end": 4, "token_start": 4, "token_end": 4, "label": "e"},
        {"start": 5, "end": 5, "token_start": 5, "token_end": 5, "label": "f"},
        {"start": 6, "end": 6, "token_start": 6, "token_end": 6, "label": "g"},
    ]

    doc["tokens"] = [
        {"text": "A", "start": 0, "end": 0, "id": 0},
        {"text": "B", "start": 1, "end": 1, "id": 1},
        {"text": "C", "start": 2, "end": 2, "id": 2},
        {"text": "D", "start": 3, "end": 3, "id": 3},
        {"text": "E", "start": 4, "end": 4, "id": 4},
        {"text": "F", "start": 5, "end": 5, "id": 5},
        {"text": "G", "start": 6, "end": 6, "id": 6},
    ]

    docs = [doc, doc, doc]

    out = [
        ("A", "a"),
        ("B", "b"),
        ("C", "c"),
        ("D", "d"),
        ("E", "e"),
        ("F", "f"),
        ("G", "g"),
        ("A", "a"),
        ("B", "b"),
        ("C", "c"),
        ("D", "d"),
        ("E", "e"),
        ("F", "f"),
        ("G", "g"),
        ("A", "a"),
        ("B", "b"),
        ("C", "c"),
        ("D", "d"),
        ("E", "e"),
        ("F", "f"),
        ("G", "g"),
    ]

    tlp = TokenLabelPairs(line_limit=73, respect_line_endings=True,
        respect_doc_endings=False)
    after = tlp.run(docs)

    assert after == out


def test_TokenLabelPairs_works_on_unlabelled():

    doc = dict()

    doc["tokens"] = [
        {"text": "A", "start": 0, "end": 0, "id": 0},
        {"text": "B", "start": 1, "end": 1, "id": 1},
        {"text": "C", "start": 2, "end": 2, "id": 2},
        {"text": "D", "start": 3, "end": 3, "id": 3},
        {"text": "E", "start": 4, "end": 4, "id": 4},
        {"text": "F", "start": 5, "end": 5, "id": 5},
        {"text": "G", "start": 6, "end": 6, "id": 6},
    ]

    docs = [doc, doc, doc]

    out = [
        ("A", None),
        ("B", None),
        ("C", None),
        ("D", None),
        ("E", None),
        ("F", None),
        ("G", None),
        (None, None),
        ("A", None),
        ("B", None),
        ("C", None),
        ("D", None),
        ("E", None),
        ("F", None),
        ("G", None),
        (None, None),
        ("A", None),
        ("B", None),
        ("C", None),
        ("D", None),
        ("E", None),
        ("F", None),
        ("G", None),
        (None, None),
    ]

    tlp = TokenLabelPairs(line_limit=73, respect_line_endings=True)
    after = tlp.run(docs)

    assert after == out


def test_TokenLabelPairs_cleans_whitespace():

    doc = dict()

    doc["tokens"] = [
        {"text": "A ", "start": 0, "end": 0, "id": 0},
        {"text": "B  ", "start": 1, "end": 1, "id": 1},
        {"text": "C   ", "start": 2, "end": 2, "id": 2},
        {"text": "D\t", "start": 3, "end": 3, "id": 3},
        {"text": "E\t\t", "start": 4, "end": 4, "id": 4},
        {"text": "F\t\t \t", "start": 5, "end": 5, "id": 5},
        {"text": "G \t \t \t \t", "start": 6, "end": 6, "id": 6},
        {"text": "\n", "start": 7, "end": 7, "id": 7},
        {"text": "\n ", "start": 8, "end": 8, "id": 8},
        {"text": "\n\t \t \t \t", "start": 9, "end": 6, "id": 9},
    ]

    docs = [doc]

    out = [
        ("A", None),
        ("B", None),
        ("C", None),
        ("D", None),
        ("E", None),
        ("F", None),
        ("G", None),
        (None, None),
        (None, None),
        (None, None),
    ]

    tlp = TokenLabelPairs(line_limit=73, respect_line_endings=True)
    after = tlp.run(docs)

    assert after == out


def test_TokenLabelPairs_retains_line_endings():
    """
    Rodrigues et al. retain the line endings as they appear in the text, meaning
    that on average a line is very short.
    """

    doc = dict()

    doc["tokens"] = [
        {"text": "\n", "start": 0, "end": 0, "id": 0},
        {"text": "\n", "start": 1, "end": 1, "id": 1},
        {"text": "\n", "start": 2, "end": 2, "id": 2},
        {"text": "\n", "start": 3, "end": 3, "id": 3},
    ]

    docs = [doc]

    out = [
        (None, None),
        (None, None),
        (None, None),
        (None, None),
    ]

    tlp = TokenLabelPairs(respect_line_endings=True)
    after = tlp.run(docs)

    assert after == out


def test_TokenLabelPairs_ignores_line_endings():

    doc = dict()

    doc["tokens"] = [
        {"text": "a", "start": 0, "end": 0, "id": 0},
        {"text": "b", "start": 1, "end": 1, "id": 1},
        {"text": "c", "start": 2, "end": 2, "id": 2},
        {"text": "d", "start": 3, "end": 3, "id": 3},
    ]

    docs = [doc]

    out = [
        ("a", None),
        ("b", None),
        (None, None),
        ("c", None),
        ("d", None),
        (None, None),
    ]

    tlp = TokenLabelPairs(line_limit=2, respect_line_endings=False)
    after = tlp.run(docs)

    assert after == out

def test_TokenLabelPairs_respects_ignores_doc_endings():

    doc = dict()

    doc["tokens"] = [
        {"text": "A", "start": 0, "end": 0, "id": 0},
        {"text": "B", "start": 1, "end": 1, "id": 1},
        {"text": "C", "start": 2, "end": 2, "id": 2},
        {"text": "D", "start": 3, "end": 3, "id": 3},
        {"text": "E", "start": 4, "end": 4, "id": 4},
        {"text": "F", "start": 5, "end": 5, "id": 5},
        {"text": "G", "start": 6, "end": 6, "id": 6},
    ]

    docs = [doc, doc, doc]

    out = [
        ("A", None),
        ("B", None),
        ("C", None),
        ("D", None),
        ("E", None),
        ("F", None),
        ("G", None),
        ("A", None),
        ("B", None),
        ("C", None),
        ("D", None),
        ("E", None),
        ("F", None),
        ("G", None),
        ("A", None),
        ("B", None),
        ("C", None),
        ("D", None),
        ("E", None),
        ("F", None),
        ("G", None),
    ]

    tlp = TokenLabelPairs(line_limit=73, respect_line_endings=False, respect_doc_endings=False)
    after = tlp.run(docs)

    assert after == out

