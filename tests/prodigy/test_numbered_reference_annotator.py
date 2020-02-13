#!/usr/bin/env python3
# coding: utf-8

import pytest
import spacy
from wellcomeml.prodigy.numbered_reference_annotator import NumberedReferenceAnnotator

@pytest.fixture(scope="function")
def nra():
    return NumberedReferenceAnnotator()


def test_numbered_reference_splitter(nra):

    numbered_reference = {
        "text": "References\n 1. \n Global update on the health sector response to HIV, 2014. Geneva: World Health Organization; \n 2014:168. \n 2. \n WHO, UNICEF, UNAIDS. Global update on HIV treatment 2013: results, impact and \n opportunities. Geneva: World Health Organization; 2013:126. \n 3. \n Consolidated guidelines on the use of antiretroviral drugs for treating and preventing HIV infection: \n recommendations for a public health approach. Geneva: World Health Organization; 2013:272. \n 4.",
        "tokens": [
            {"text": "References", "start": 0, "end": 10, "id": 0},
            {"text": "\n ", "start": 10, "end": 12, "id": 1},
            {"text": "1", "start": 12, "end": 13, "id": 2},
            {"text": ".", "start": 13, "end": 14, "id": 3},
            {"text": "\n ", "start": 15, "end": 17, "id": 4},
            {"text": "Global", "start": 17, "end": 23, "id": 5},
            {"text": "update", "start": 24, "end": 30, "id": 6},
            {"text": "on", "start": 31, "end": 33, "id": 7},
            {"text": "the", "start": 34, "end": 37, "id": 8},
            {"text": "health", "start": 38, "end": 44, "id": 9},
            {"text": "sector", "start": 45, "end": 51, "id": 10},
            {"text": "response", "start": 52, "end": 60, "id": 11},
            {"text": "to", "start": 61, "end": 63, "id": 12},
            {"text": "HIV", "start": 64, "end": 67, "id": 13},
            {"text": ",", "start": 67, "end": 68, "id": 14},
            {"text": "2014", "start": 69, "end": 73, "id": 15},
            {"text": ".", "start": 73, "end": 74, "id": 16},
            {"text": "Geneva", "start": 75, "end": 81, "id": 17},
            {"text": ":", "start": 81, "end": 82, "id": 18},
            {"text": "World", "start": 83, "end": 88, "id": 19},
            {"text": "Health", "start": 89, "end": 95, "id": 20},
            {"text": "Organization", "start": 96, "end": 108, "id": 21},
            {"text": ";", "start": 108, "end": 109, "id": 22},
            {"text": "\n ", "start": 110, "end": 112, "id": 23},
            {"text": "2014:168", "start": 112, "end": 120, "id": 24},
            {"text": ".", "start": 120, "end": 121, "id": 25},
            {"text": "\n ", "start": 122, "end": 124, "id": 26},
            {"text": "2", "start": 124, "end": 125, "id": 27},
            {"text": ".", "start": 125, "end": 126, "id": 28},
            {"text": "\n ", "start": 127, "end": 129, "id": 29},
            {"text": "WHO", "start": 129, "end": 132, "id": 30},
            {"text": ",", "start": 132, "end": 133, "id": 31},
            {"text": "UNICEF", "start": 134, "end": 140, "id": 32},
            {"text": ",", "start": 140, "end": 141, "id": 33},
            {"text": "UNAIDS", "start": 142, "end": 148, "id": 34},
            {"text": ".", "start": 148, "end": 149, "id": 35},
            {"text": "Global", "start": 150, "end": 156, "id": 36},
            {"text": "update", "start": 157, "end": 163, "id": 37},
            {"text": "on", "start": 164, "end": 166, "id": 38},
            {"text": "HIV", "start": 167, "end": 170, "id": 39},
            {"text": "treatment", "start": 171, "end": 180, "id": 40},
            {"text": "2013", "start": 181, "end": 185, "id": 41},
            {"text": ":", "start": 185, "end": 186, "id": 42},
            {"text": "results", "start": 187, "end": 194, "id": 43},
            {"text": ",", "start": 194, "end": 195, "id": 44},
            {"text": "impact", "start": 196, "end": 202, "id": 45},
            {"text": "and", "start": 203, "end": 206, "id": 46},
            {"text": "\n ", "start": 207, "end": 209, "id": 47},
            {"text": "opportunities", "start": 209, "end": 222, "id": 48},
            {"text": ".", "start": 222, "end": 223, "id": 49},
            {"text": "Geneva", "start": 224, "end": 230, "id": 50},
            {"text": ":", "start": 230, "end": 231, "id": 51},
            {"text": "World", "start": 232, "end": 237, "id": 52},
            {"text": "Health", "start": 238, "end": 244, "id": 53},
            {"text": "Organization", "start": 245, "end": 257, "id": 54},
            {"text": ";", "start": 257, "end": 258, "id": 55},
            {"text": "2013:126", "start": 259, "end": 267, "id": 56},
            {"text": ".", "start": 267, "end": 268, "id": 57},
            {"text": "\n ", "start": 269, "end": 271, "id": 58},
            {"text": "3", "start": 271, "end": 272, "id": 59},
            {"text": ".", "start": 272, "end": 273, "id": 60},
            {"text": "\n ", "start": 274, "end": 276, "id": 61},
            {"text": "Consolidated", "start": 276, "end": 288, "id": 62},
            {"text": "guidelines", "start": 289, "end": 299, "id": 63},
            {"text": "on", "start": 300, "end": 302, "id": 64},
            {"text": "the", "start": 303, "end": 306, "id": 65},
            {"text": "use", "start": 307, "end": 310, "id": 66},
            {"text": "of", "start": 311, "end": 313, "id": 67},
            {"text": "antiretroviral", "start": 314, "end": 328, "id": 68},
            {"text": "drugs", "start": 329, "end": 334, "id": 69},
            {"text": "for", "start": 335, "end": 338, "id": 70},
            {"text": "treating", "start": 339, "end": 347, "id": 71},
            {"text": "and", "start": 348, "end": 351, "id": 72},
            {"text": "preventing", "start": 352, "end": 362, "id": 73},
            {"text": "HIV", "start": 363, "end": 366, "id": 74},
            {"text": "infection", "start": 367, "end": 376, "id": 75},
            {"text": ":", "start": 376, "end": 377, "id": 76},
            {"text": "\n ", "start": 378, "end": 380, "id": 77},
            {"text": "recommendations", "start": 380, "end": 395, "id": 78},
            {"text": "for", "start": 396, "end": 399, "id": 79},
            {"text": "a", "start": 400, "end": 401, "id": 80},
            {"text": "public", "start": 402, "end": 408, "id": 81},
            {"text": "health", "start": 409, "end": 415, "id": 82},
            {"text": "approach", "start": 416, "end": 424, "id": 83},
            {"text": ".", "start": 424, "end": 425, "id": 84},
            {"text": "Geneva", "start": 426, "end": 432, "id": 85},
            {"text": ":", "start": 432, "end": 433, "id": 86},
            {"text": "World", "start": 434, "end": 439, "id": 87},
            {"text": "Health", "start": 440, "end": 446, "id": 88},
            {"text": "Organization", "start": 447, "end": 459, "id": 89},
            {"text": ";", "start": 459, "end": 460, "id": 90},
            {"text": "2013:272", "start": 461, "end": 469, "id": 91},
            {"text": ".", "start": 469, "end": 470, "id": 92},
            {"text": "\n", "start": 470, "end": 471, "id": 92},
            {"text": "3", "start": 471, "end": 472, "id": 92},
            {"text": ".", "start": 472, "end": 473, "id": 92},
    ]
    }

    docs = list(nra.run([numbered_reference]))
    text = docs[0]["text"]
    spans = docs[0]["spans"]
    ref_1 = text[spans[0]["start"]:spans[0]["end"]]
    ref_2 = text[spans[1]["start"]:spans[1]["end"]]
    ref_3 = text[spans[2]["start"]:spans[2]["end"]]

    assert len(spans) == 3
    assert ref_1 == "Global update on the health sector response to HIV, 2014. Geneva: World Health Organization; \n 2014:168."
    assert ref_2.strip() == "WHO, UNICEF, UNAIDS. Global update on HIV treatment 2013: results, impact and \n opportunities. Geneva: World Health Organization; 2013:126."
    assert ref_3.strip() == "Consolidated guidelines on the use of antiretroviral drugs for treating and preventing HIV infection: \n recommendations for a public health approach. Geneva: World Health Organization; 2013:272."

def test_numbered_reference_splitter_line_endings(nra):
    """
    Test case where there two line enedings immediately preceding the reference
    index.
    """

    numbered_reference = {
        "text": "References\n\n1. \n Global update on the health sector response to HIV, 2014. Geneva: World Health Organization; \n 2014:168. \n\n2. \n WHO, UNICEF, UNAIDS. Global update on HIV treatment 2013: results, impact and \n opportunities. Geneva: World Health Organization; 2013:126.\n\n3.",
        "tokens": [
            {"text": "References", "start": 0, "end": 10, "id": 0},
            {"text": "\n\n", "start": 10, "end": 12, "id": 1},
            {"text": "1", "start": 12, "end": 13, "id": 2},
            {"text": ".", "start": 13, "end": 14, "id": 3},
            {"text": "\n ", "start": 15, "end": 17, "id": 4},
            {"text": "Global", "start": 17, "end": 23, "id": 5},
            {"text": "update", "start": 24, "end": 30, "id": 6},
            {"text": "on", "start": 31, "end": 33, "id": 7},
            {"text": "the", "start": 34, "end": 37, "id": 8},
            {"text": "health", "start": 38, "end": 44, "id": 9},
            {"text": "sector", "start": 45, "end": 51, "id": 10},
            {"text": "response", "start": 52, "end": 60, "id": 11},
            {"text": "to", "start": 61, "end": 63, "id": 12},
            {"text": "HIV", "start": 64, "end": 67, "id": 13},
            {"text": ",", "start": 67, "end": 68, "id": 14},
            {"text": "2014", "start": 69, "end": 73, "id": 15},
            {"text": ".", "start": 73, "end": 74, "id": 16},
            {"text": "Geneva", "start": 75, "end": 81, "id": 17},
            {"text": ":", "start": 81, "end": 82, "id": 18},
            {"text": "World", "start": 83, "end": 88, "id": 19},
            {"text": "Health", "start": 89, "end": 95, "id": 20},
            {"text": "Organization", "start": 96, "end": 108, "id": 21},
            {"text": ";", "start": 108, "end": 109, "id": 22},
            {"text": "\n ", "start": 110, "end": 112, "id": 23},
            {"text": "2014:168", "start": 112, "end": 120, "id": 24},
            {"text": ".", "start": 120, "end": 121, "id": 25},
            {"text": "\n\n", "start": 122, "end": 124, "id": 26},
            {"text": "2", "start": 124, "end": 125, "id": 27},
            {"text": ".", "start": 125, "end": 126, "id": 28},
            {"text": "\n ", "start": 127, "end": 129, "id": 29},
            {"text": "WHO", "start": 129, "end": 132, "id": 30},
            {"text": ",", "start": 132, "end": 133, "id": 31},
            {"text": "UNICEF", "start": 134, "end": 140, "id": 32},
            {"text": ",", "start": 140, "end": 141, "id": 33},
            {"text": "UNAIDS", "start": 142, "end": 148, "id": 34},
            {"text": ".", "start": 148, "end": 149, "id": 35},
            {"text": "Global", "start": 150, "end": 156, "id": 36},
            {"text": "update", "start": 157, "end": 163, "id": 37},
            {"text": "on", "start": 164, "end": 166, "id": 38},
            {"text": "HIV", "start": 167, "end": 170, "id": 39},
            {"text": "treatment", "start": 171, "end": 180, "id": 40},
            {"text": "2013", "start": 181, "end": 185, "id": 41},
            {"text": ":", "start": 185, "end": 186, "id": 42},
            {"text": "results", "start": 187, "end": 194, "id": 43},
            {"text": ",", "start": 194, "end": 195, "id": 44},
            {"text": "impact", "start": 196, "end": 202, "id": 45},
            {"text": "and", "start": 203, "end": 206, "id": 46},
            {"text": "\n ", "start": 207, "end": 209, "id": 47},
            {"text": "opportunities", "start": 209, "end": 222, "id": 48},
            {"text": ".", "start": 222, "end": 223, "id": 49},
            {"text": "Geneva", "start": 224, "end": 230, "id": 50},
            {"text": ":", "start": 230, "end": 231, "id": 51},
            {"text": "World", "start": 232, "end": 237, "id": 52},
            {"text": "Health", "start": 238, "end": 244, "id": 53},
            {"text": "Organization", "start": 245, "end": 257, "id": 54},
            {"text": ";", "start": 257, "end": 258, "id": 55},
            {"text": "2013:126", "start": 259, "end": 267, "id": 56},
            {"text": ".", "start": 260, "end": 261, "id": 57},
            {"text": "\n\n", "start": 261, "end": 263, "id": 58},
            {"text": "3", "start": 262, "end": 264, "id": 59},
            {"text": ".", "start": 263, "end": 265, "id": 60},
    ]
    }

    docs = list(nra.run([numbered_reference]))
    text = docs[0]["text"]
    spans = docs[0]["spans"]
    ref_1 = text[spans[0]["start"]:spans[0]["end"]]
    ref_2 = text[spans[1]["start"]:spans[1]["end"]]

    assert len(spans) == 2
    assert ref_1.strip() == "Global update on the health sector response to HIV, 2014. Geneva: World Health Organization; \n 2014:168."
    assert ref_2.strip() == "WHO, UNICEF, UNAIDS. Global update on HIV treatment 2013: results, impact and \n opportunities. Geneva: World Health Organization; 2013:126"
