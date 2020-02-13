#!/usr/bin/env python3
# coding: utf-8

import pytest
from wellcomeml.prodigy.reach_to_prodigy import ReachToProdigy

@pytest.fixture(scope="function")
def stp():
    ref_sections = [{}, {}, {}]
    return ReachToProdigy(ref_sections)

def test_combine_n_rows(stp):

    doc = list(range(100, 200))
    out = stp.combine_n_rows(doc, n=5, join_char=" ")

    last_in_doc = doc[len(doc) -1]
    last_in_out = int(out[-1].split(" ")[-1])

    assert last_in_doc == last_in_out

    assert out[0] == '100 101 102 103 104'
    assert out[-2] == '190 191 192 193 194'
    assert out[-1] == '195 196 197 198 199'

def test_combine_n_rows_uneven_split(stp):

    doc = list(range(100, 200))
    out = stp.combine_n_rows(doc, n=7, join_char=" ")

    last_in_doc = doc[len(doc) -1]
    last_in_out = int(out[-1].split(" ")[-1])

    assert last_in_doc == last_in_out
    assert len(out[-1].split(" ")) == 2
    assert len(out[-2].split(" ")) == 7

    assert out[0] == '100 101 102 103 104 105 106'
    assert out[-2] == '191 192 193 194 195 196 197'
    assert out[-1] == '198 199'
