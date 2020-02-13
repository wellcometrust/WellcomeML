#!/usr/bin/env python3
# coding: utf-8

import itertools

import plac

from ..io import read_jsonl, write_jsonl
from ..logger import logger


class TokenTagger:
    """
    Converts data in prodigy format with full reference spans to per-token spans

    Expects one of four lables for the spans:

    * BE: A complete reference
    * BI: A frgament of reference that captures the beginning but not the end
    * IE: A frgament of reference that captures the end but not the beginning
    * II: A fragment of a reference that captures neither the beginning nor the
        end .
    """

    def __init__(self):

        self.out = []

    def tag_doc(self, doc):
        """
        Tags a document with the appropriate labels

        Args:
            doc(dict): A single document in prodigy dict format to be labelled.
        """

        bie_spans = self.reference_spans(doc["spans"], doc["tokens"])
        o_spans = self.outside_spans(bie_spans, doc["tokens"])

        # Flatten into one list.

        spans = itertools.chain(bie_spans, o_spans)

        # Sort by token id to ensure it is ordered.

        spans = sorted(spans, key=lambda k: k['token_start'])

        doc["spans"] = spans

        return doc

    def run(self, docs):
        """
        Main class method for tagging multiple documents.

        Args:
            docs(dict): A list of docs in prodigy dict format to be labelled.
        """

        for doc in docs:

            self.out.append(self.tag_doc(doc))

        return self.out

    def reference_spans(self, spans, tokens):
        """
        Given a whole reference span as labelled in prodigy, break this into
        appropriate single token spans depending on the label that was applied to
        the whole reference span.
        """
        split_spans = []

        for span in spans:
            if span["label"] in ["BE", "be"]:

                split_spans.extend(
                    self.split_long_span(tokens, span, "b-r", "e-r")
                )

            elif span["label"] in ["BI", "bi"]:

                split_spans.extend(
                    self.split_long_span(tokens, span, "b-r", "i-r")
                )

            elif span["label"] in ["IE", "ie"]:

                split_spans.extend(
                    self.split_long_span(tokens, span, "i-r", "e-r")
                )

            elif span["label"] in ["II", "ii"]:

                split_spans.extend(
                    self.split_long_span(tokens, span, "i-r", "i-r")
                )

        return split_spans


    def outside_spans(self, spans, tokens):
        """
        Label tokens with `o` if they are outside a reference

        Args:
            spans(list): Spans in prodigy format.
            tokens(list): Tokens in prodigy format.

        Returns:
            list: A list of spans in prodigy format that comprises the tokens which
                are outside of a reference.
        """
        # Get the diff between inside and outside tokens

        span_indices = set([span["token_start"] for span in spans])
        token_indices = set([token["id"] for token in tokens])

        outside_indices = token_indices - span_indices

        outside_spans = []

        for index in outside_indices:
            outside_spans.append(self.create_span(tokens, index, "o"))

        return outside_spans


    def create_span(self, tokens, index, label):
        """
        Given a list of tokens, (in prodigy format) and an index relating to one of
        those tokens, and a new label: create a single token span using the new
        label, and the token selected by `index`.
        """

        token = tokens[index]

        span = {
            "start": token["start"],
            "end": token["end"],
            "token_start": token["id"],
            "token_end": token["id"],
            "label": label,
        }

        return span


    def split_long_span(self, tokens, span, start_label, end_label):
        """
        Split a milti-token span into `n` spans of lengh `1`, where `n=len(tokens)`
        """

        spans = []
        spans.append(self.create_span(tokens, span["token_start"], start_label))
        spans.append(self.create_span(tokens, span["token_end"], end_label))

        for index in range(span["token_start"] + 1, span["token_end"]):
            spans.append(self.create_span(tokens, index, "i-r"))

        spans = sorted(spans, key=lambda k: k['token_start'])

        return spans

@plac.annotations(
    input_file=(
        "Path to jsonl file containing chunks of references in prodigy format.",
        "positional",
        None,
        str
    ),
    output_file=(
        "Path to jsonl file into which fully annotate files will be saved.",
        "positional",
        None,
        str
    )
)

def reference_to_token_annotations(input_file, output_file):
    """ Converts a file output by prodigy (using prodigy db-out) from
    references level annotations to individual level annotations. The rationale
    for this is that reference level annotations are much easier for humans to
    do, but not useful when training a token level model.

    This function is predominantly useful fot tagging reference spans, but may
    also have a function with other references annotations.
    """

    partially_annotated = read_jsonl(input_file)

    # Only run the tagger on annotated examples.

    partially_annotated = [doc for doc in partially_annotated if doc.get("spans")]

    logger.info("Loaded %s documents with reference annotations", len(partially_annotated))

    annotator = TokenTagger(partially_annotated)

    fully_annotated = annotator.run()

    write_jsonl(fully_annotated, output_file=output_file)

    logger.info("Fully annotated references written to %s", output_file)
