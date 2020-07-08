#!/usr/bin/env python3
# coding: utf-8

import en_core_web_sm
import pytest

from wellcomeml.spacy.spacy_doc_to_prodigy import SpacyDocToProdigy


@pytest.fixture(scope="module")
def nlp():
    return en_core_web_sm.load()


def test_return_one_prodigy_doc_fails_if_passed_wrong_type():

    with pytest.raises(TypeError):
        wrong_format = [
            "this is the text",
            {"entities": [[0, 1, "PERSON"], [2, 4, "COMPANY"]]},
        ]

        spacy_to_prodigy = SpacyDocToProdigy()
        spacy_to_prodigy.return_one_prodigy_doc(wrong_format)


def test_SpacyDocToProdigy(nlp):

    # https://www.theguardian.com/world/2019/oct/30/pinochet-economic-model-current-crisis-chile
    before = nlp("After 12 days of mass demonstrations, rioting and human rights violations,"
                 " the government of President Sebastián Piñera must now find a way out of the"
                 " crisis that has engulfed Chile.")

    stp = SpacyDocToProdigy()
    actual = stp.run([before])

    expected = [
        {
            'text': 'After 12 days of mass demonstrations, rioting and human rights violations,'
                    ' the government of President Sebastián Piñera must now find a way out of the'
                    ' crisis that has engulfed Chile.',
            'spans': [
                {'token_start': 1, 'token_end': 3, 'start': 6, 'end': 13, 'label': 'DATE'},
                {'token_start': 17, 'token_end': 19, 'start': 103, 'end': 119, 'label': 'PERSON'},
                {'token_start': 31, 'token_end': 32, 'start': 176, 'end': 181, 'label': 'GPE'}
            ],
            'tokens': [
                {'text': 'After', 'start': 0, 'end': 5, 'id': 0},
                {'text': '12', 'start': 6, 'end': 8, 'id': 1},
                {'text': 'days', 'start': 9, 'end': 13, 'id': 2},
                {'text': 'of', 'start': 14, 'end': 16, 'id': 3},
                {'text': 'mass', 'start': 17, 'end': 21, 'id': 4},
                {'text': 'demonstrations', 'start': 22, 'end': 36, 'id': 5},
                {'text': ',', 'start': 36, 'end': 37, 'id': 6},
                {'text': 'rioting', 'start': 38, 'end': 45, 'id': 7},
                {'text': 'and', 'start': 46, 'end': 49, 'id': 8},
                {'text': 'human', 'start': 50, 'end': 55, 'id': 9},
                {'text': 'rights', 'start': 56, 'end': 62, 'id': 10},
                {'text': 'violations', 'start': 63, 'end': 73, 'id': 11},
                {'text': ',', 'start': 73, 'end': 74, 'id': 12},
                {'text': 'the', 'start': 75, 'end': 78, 'id': 13},
                {'text': 'government', 'start': 79, 'end': 89, 'id': 14},
                {'text': 'of', 'start': 90, 'end': 92, 'id': 15},
                {'text': 'President', 'start': 93, 'end': 102, 'id': 16},
                {'text': 'Sebastián', 'start': 103, 'end': 112, 'id': 17},
                {'text': 'Piñera', 'start': 113, 'end': 119, 'id': 18},
                {'text': 'must', 'start': 120, 'end': 124, 'id': 19},
                {'text': 'now', 'start': 125, 'end': 128, 'id': 20},
                {'text': 'find', 'start': 129, 'end': 133, 'id': 21},
                {'text': 'a', 'start': 134, 'end': 135, 'id': 22},
                {'text': 'way', 'start': 136, 'end': 139, 'id': 23},
                {'text': 'out', 'start': 140, 'end': 143, 'id': 24},
                {'text': 'of', 'start': 144, 'end': 146, 'id': 25},
                {'text': 'the', 'start': 147, 'end': 150, 'id': 26},
                {'text': 'crisis', 'start': 151, 'end': 157, 'id': 27},
                {'text': 'that', 'start': 158, 'end': 162, 'id': 28},
                {'text': 'has', 'start': 163, 'end': 166, 'id': 29},
                {'text': 'engulfed', 'start': 167, 'end': 175, 'id': 30},
                {'text': 'Chile', 'start': 176, 'end': 181, 'id': 31},
                {'text': '.', 'start': 181, 'end': 182, 'id': 32}
            ]
        }
    ]

    assert expected == actual
