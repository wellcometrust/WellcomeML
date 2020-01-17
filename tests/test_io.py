#!/usr/bin/env python3
# coding: utf-8

import os
import tempfile

import pytest

from wellcomeml.io import read_jsonl, write_jsonl

from .common import TEST_JSONL


@pytest.fixture(scope="module")
def temp_file():
    temp_file, temp_file_name = tempfile.mkstemp()

    return temp_file_name

def test_read_jsonl():

    expected = [{
        "text": "a b c\n a b c",
        "tokens": [
            {'text': 'a', 'start': 0, 'end': 1, 'id': 0},
            {'text': 'b', 'start': 2, 'end': 3, 'id': 1},
            {'text': 'c', 'start': 4, 'end': 5, 'id': 2},
            {'text': '\n ', 'start': 5, 'end': 7, 'id': 3},
            {'text': 'a', 'start': 7, 'end': 8, 'id': 4},
            {'text': 'b', 'start': 9, 'end': 10, 'id': 5},
            {'text': 'c', 'start': 11, 'end': 12, 'id': 6}
        ],
        "spans": [
            {'start': 2, 'end': 3, 'token_start': 1, "token_end": 2, "label": "b"},
            {'start': 4, 'end': 5, 'token_start': 2, "token_end": 3, "label": "i"},
            {'start': 7, 'end': 8, 'token_start': 4, "token_end": 5, "label": "i"},
            {'start': 9, 'end': 10, 'token_start': 5, "token_end": 6, "label": "e"},
        ]
    }]

    expected = expected * 3

    actual = read_jsonl(TEST_JSONL)
    assert expected == actual

def test_write_jsonl(temp_file):

    expected = [{
        "text": "a b c\n a b c",
        "tokens": [
            {'text': 'a', 'start': 0, 'end': 1, 'id': 0},
            {'text': 'b', 'start': 2, 'end': 3, 'id': 1},
            {'text': 'c', 'start': 4, 'end': 5, 'id': 2},
            {'text': '\n ', 'start': 5, 'end': 7, 'id': 3},
            {'text': 'a', 'start': 7, 'end': 8, 'id': 4},
            {'text': 'b', 'start': 9, 'end': 10, 'id': 5},
            {'text': 'c', 'start': 11, 'end': 12, 'id': 6}
        ],
        "spans": [
            {'start': 2, 'end': 3, 'token_start': 1, "token_end": 2, "label": "b"},
            {'start': 4, 'end': 5, 'token_start': 2, "token_end": 3, "label": "i"},
            {'start': 7, 'end': 8, 'token_start': 4, "token_end": 5, "label": "i"},
            {'start': 9, 'end': 10, 'token_start': 5, "token_end": 6, "label": "e"},
        ]
    }]

    expected = expected * 3

    write_jsonl(expected, temp_file)
    actual = read_jsonl(temp_file)

    assert expected == actual

    # Clean up

    if os.path.isfile(temp_file):
        os.remove(temp_file)
