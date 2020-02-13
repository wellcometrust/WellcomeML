# coding: utf-8
#!/usr/bin/env python3

import re

import plac

from ..io import read_jsonl, write_jsonl
from ..logger import logger

REGEX = r"\n{1,2}(?:(?:\s)|(?:\(|\[))?(?:\d{1,2})(?:(?:\.\)|\.\]|\]\n|\.|\s)|(?:\]|\)))(\s+)?(?:\n)?(?:\s+)?(?!Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"

class NumberedReferenceAnnotator:
    """
    Takes reference sections with numeric labelling scraped by Reach in prodigy
    format, and labels the references as spans by splitting them using regex.

    Note that you must identify numbered reference section first. This can be
    done with a simple textcat model trained in prodigy.
    """

    def __init__(self):

        self.regex = r""

    def run(self, docs, regex=REGEX):

        self.regex = regex

        for doc in docs:

            spans = self.label_numbered_references(doc["text"], doc["tokens"])
            doc["spans"] = spans

            yield doc

    def label_numbered_references(self, text, tokens):

        # Search for number reference using regex

        splits = list(re.finditer(self.regex, text))
        spans = []

        for index in range(0, len(splits) - 1):

            # Calculate the approximate start and end of the reference using
            # the character offsets returned by re.finditer.

            start = splits[index].end()
            end = splits[index + 1].start()

            # Calculate which is the closest token to the character offset
            # returned above.

            token_start = self._find_closest_token(tokens, start, "start")
            token_end = self._find_closest_token(tokens, end, "end")

            # To avoid the possibility of mismatches between the character
            # offset and the token offset, reset the character offsets
            # based on the token offsets.

            start = self._get_token_offset(tokens, token_start, "start")
            end = self._get_token_offset(tokens, token_end, "end")

            # Create dict and append

            span = {
                "start": start,
                "end": end,
                "token_start": token_start,
                "token_end": token_end,
                "label": "BE"
            }

            spans.append(span)

        return spans


    def _find_closest_token(self, tokens, char_offset, pos_string):
        """
        Find the token start/end closest to "number"

        Args:
            tokens: A list of token dicts from a prodigy document.
            char_offset(int): A character offset relating to either the start or the
                end of a token.
            pos_string(str): One of ["start", "end"] denoting whether `char_offset`
                is a start or the end of a token
        """
        token_map = self._token_start_mapper(tokens, pos_string)
        token_key = self._find_closest_number(token_map.keys(), char_offset)

        return token_map[token_key]

    def _get_token_offset(self, tokens, token_id, pos_string):
        """
        Return the character offset for the token with id == token_id
        """

        token_match = (token[pos_string] for token in tokens if token["id"] == token_id)

        return next(token_match, None)

    def _find_closest_number(self, numbers, number):
        """ Find the closest match in a list of numbers when presented with
        a number
        """

        return min(numbers, key=lambda x:abs(x - number))

    def _token_start_mapper(self, tokens, pos_string):
        """ Map token id by the token start/end position
        """

        return {token[pos_string]:token["id"] for token in tokens}


@plac.annotations(
    input_file=(
        "Path to jsonl file containing numbered reference sections as docs.",
        "positional",
        None,
        str
    ),
    output_file=(
        "Path to output jsonl file containing prodigy docs with numbered references labelled.",
        "positional",
        None,
        str
    )
)
def annotate_numbered_references(input_file, output_file):
    """
    Takes reference sections with numeric labelling scraped by Reach in prodigy
    format, and labels the references as spans by splitting them using regex.
    """

    numbered_reference_sections = read_jsonl(input_file)

    logger.info("Loaded %s prodigy docs", len(numbered_reference_sections))

    nra = NumberedReferenceAnnotator()
    docs = list(nra.run[numbered_reference_sections])

    write_jsonl(output_file)

    logger.info("Wrote %s annotated references to %s", len(docs),
        output_file)
