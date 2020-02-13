#!/usr/bin/env python3
# coding: utf-8

"""
Class used in scripts/prodigy_to_tsv.py which converts token annotated jsonl
files to tab-separated-values files for use in the deep reference parser
"""

import csv
import re

import numpy as np
import plac

from ..io import read_jsonl

from ..logger import logger


class TokenLabelPairs:
    """
    Convert prodigy format docs or list of lists into tuples of (token, label).
    """

    def __init__(self, line_limit=73, respect_line_endings=True, respect_doc_endings=True):
        """
        Args:
            line_limit(int): Maximum number of tokens allowed per training
                example. If you are planning to use this data for making
                predictions, then this should correspond to the max_words
                attribute for the DeepReferenceParser class used to train the
                model.
            respect_line_endings(bool): If true, line endings appearing in the
                text will be respected, leading to much shorter line lengths
                usually <10. Typically this results in a much worser performing
                model, but follows the convention set by Rodrigues et al.
            respect_doc_endings(bool): If true, a line ending is added at the
                end of each document. If false, then the end of a document flows
                into the beginning of the next document.
        """

        self.line_count = 0
        self.line_lengths = []
        self.line_limit = line_limit
        self.respect_doc_endings = respect_doc_endings
        self.respect_line_endings = respect_line_endings

    def run(self, docs):
        """
        """

        out = []

        for doc in docs:
            out.extend(self.yield_token_label_pair(doc))

        self.stats(out)

        return out


    def stats(self, out):

        avg_line_len = np.round(np.mean(self.line_lengths), 2)

        logger.debug("Returning %s examples", self.line_count)
        logger.debug("Average line length: %s", avg_line_len)

    def yield_token_label_pair(self, doc, lists=False):
        """
        Expect list of jsons loaded from a jsonl

        Args:
            doc (dict): Document in prodigy format or list of lists
            lists (bool): Expect a list of lists rather than a prodigy format
                dict?

        NOTE: Makes the assumption that every token has been labelled in spans. This
        assumption will be true if the data has been labelled with prodigy, then
        spans covering entire references have been converted to token spans. OR that
        there are no spans at all, and this is being used to prepare data for
        prediction.
        """

        # Ensure that spans and tokens are sorted (they should be)

        if lists:
            tokens = doc
        else:
            tokens = sorted(doc["tokens"], key=lambda k: k["id"])

        # For prediction, documents may not yet have spans. If they do, sort
        # them too based on token_start which is equivalent to id in
        # doc["tokens"].

        spans = doc.get("spans")

        if spans:
            spans = sorted(doc["spans"], key=lambda k: k["token_start"])

        # Set a token counter that is used to limit the number of tokens to
        # line_limit.

        token_counter = int(0)

        doc_len = len(tokens)

        for i, token in enumerate(tokens, 1):

            label = None

            # For case when tokens have been labelled with spans (for training
            # data).

            if spans:
                # Need to remove one from index as it starts at 1!
                label = spans[i - 1].get("label")

            text = token["text"]

            # If the token is empty even if it has been labelled, pass it

            if text == "":

                pass

            # If the token is a newline (and possibly other characters) and we want
            # to respect line endings in the text, then yield a (None, None) tuple
            # which will be converted to a blank line when the resulting tsv file
            # is read.

            elif re.search(r"\n", text) and self.respect_line_endings:

                # Is it blank after whitespace is removed?

                if text.strip() == "":

                    yield (None, None)

                self.line_lengths.append(token_counter)
                self.line_count += 1

                token_counter = 0

            elif token_counter == self.line_limit:

                # Yield None, None to signify a line ending, then yield the next
                # token.

                yield (None, None)
                yield (text.strip(), label)

                # Set to one to account for the first token being added.

                self.line_lengths.append(token_counter)
                self.line_count += 1

                token_counter = 1

            elif i == doc_len and self.respect_doc_endings:

                # Case when the end of the document has been reached, but it is
                # less than self.lime_limit. This assumes that we want to retain
                # a line ending which denotes the end of a document, and the
                # start of new one.

                yield (text.strip(), label)
                yield (None, None)

                self.line_lengths.append(token_counter)
                self.line_count += 1

            else:

                # Returned the stripped label.

                yield (text.strip(), label)

                token_counter += 1


@plac.annotations(
    input_file=(
        "Path to jsonl file containing prodigy docs.",
        "positional",
        None,
        str
    ),
    output_file=(
        "Path to output tsv file.",
        "positional",
        None,
        str
    )
)
def prodigy_to_tsv(input_file, output_file):
    """
    Convert token annotated jsonl to token annotated tsv ready for use in the
    Rodrigues model.
    """

    annotated_data = read_jsonl(input_file)

    logger.info("Loaded %s prodigy docs", len(annotated_data))

    tlp = TokenLabelPairs()
    token_label_pairs = list(tlp.run(annotated_data))

    with open(output_file, 'w') as fb:
        writer = csv.writer(fb, delimiter="\t")
        # Write DOCSTART and a blank line
        writer.writerows([("DOCSTART", None), (None, None)])
        writer.writerows(token_label_pairs)

    logger.info("Wrote %s token/label pairs to %s", len(token_label_pairs),
        output_file)

