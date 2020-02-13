#!/usr/bin/env python3
# coding: utf-8

import copy
import itertools

import en_core_web_sm as model
import plac

from ..io import read_jsonl, write_jsonl
from ..logger import logger


class ReachToProdigy:
    """
    Converts json of scraped reference section into prodigy style json.

    The resulting json can then be loaded into prodigy is required.

    Expects dict in the following format:

    ```
    {
    ...,
    "sections": {
            "Reference": "References\n1. Upson. M. (2018) ..."
        }
    }

    ```

    Returns references in the following format:

    ```
    [{
        'text': ' This is an example with a linebreak\n',
        'meta': {'doc_hash': None, 'provider': None, 'line_number': 3},
        'tokens': [
            {'text': ' ', 'start': 0, 'end': 1, 'id': 0},
            {'text': 'This', 'start': 1, 'end': 5, 'id': 1},
            {'text': 'is', 'start': 6, 'end': 8, 'id': 2},
            {'text': 'an', 'start': 9, 'end': 11, 'id': 3},
            {'text': 'example', 'start': 12, 'end': 19, 'id': 4},
            {'text': 'with', 'start': 20, 'end': 24, 'id': 5},
            {'text': 'a', 'start': 25, 'end': 26, 'id': 6},
            {'text': 'linebreak', 'start': 27, 'end': 36, 'id': 7},
            {'text': '\n', 'start': 36, 'end': 37, 'id': 8}]
    },
    ...
    ]

    ```
    """

    def __init__(self, ref_sections, lines=10, split_char="\n",
                 add_linebreak=True, join_char=" "):
        """
        Args:
            ref_sections(list): List of dicts extracted in scrape.
            lines(int): Number of lines to combine into one chunk
            split_char(str): Character to split lines on.
            add_linebreak(bool): Should a linebreak be re-added so that it is
                clear where a break was made?
            join_chars(str): Which character will be used to join lines at the
                point which they are merged into a chunk.
        """

        self.ref_sections = ref_sections
        self.lines = lines
        self.split_char = split_char
        self.add_linebreak = add_linebreak
        self.join_char = join_char

        self.nlp = model.load()

    def run(self):
        """
        Main method of the class
        """

        prodigy_format = []

        for i, refs in enumerate(self.ref_sections):

            one_record = self.one_record_to_prodigy_format(refs, self.nlp,
                self.lines, self.split_char, self.add_linebreak, self.join_char)

            # If something is returned (i.e. there is a ref section)
            # then append to prodigy_format.

            if one_record:

                prodigy_format.append(one_record)

        out = list(itertools.chain.from_iterable(prodigy_format))

        logger.info("Returned %s reference sections", len(out))

        return out

    def one_record_to_prodigy_format(self, input_dict, nlp, lines=10, split_char="\n",
        add_linebreak=True, join_char=" "):
        """
        Convert one dict produced by the scrape to a list of prodigy dicts

        Args:
            input_dict(dict): One reference section dict from the scrape
            nlp: A spacy model, for example loaded with spacy.load("en_core_web_sm")
            lines(int): Number of lines to combine into one chunk
            split_char(str): Character to split lines on.
            add_linebreak(bool): Should a linebreak be re-added so that it is
                clear where a break was made?
            join_chars(str): Which character will be used to join lines at the
                point which they are merged into a chunk.
        """

        out = []

        # Only continue if references are found

        if input_dict:

            sections = input_dict.get("sections")

            # If there is something in sections: this will be a keyword for example
            # reference, or bibliography, etc

            if sections:

                # In case there are more than one keyword, cycle through them

                for _, refs in sections.items():

                    # Refs will be a list, so cycle through it in case there was
                    # more than one section found with the same keyword

                    for ref in refs:

                        if refs:

                            refs_lines = self.split_lines(ref, split_char=split_char, add_linebreak=add_linebreak)
                            refs_grouped = self.combine_n_rows(refs_lines, n=lines, join_char=join_char)

                            _meta = {
                                "doc_hash": input_dict.get("file_hash"),
                                "provider": input_dict.get("provider"),
                            }

                            for i, lines in enumerate(refs_grouped):

                                meta = copy.deepcopy(_meta)

                                meta["line_number"] = i

                                tokens = nlp.tokenizer(lines)
                                formatted_tokens = [self.format_token(i) for i in tokens]

                                out.append({"text": lines, "meta": meta, "tokens": formatted_tokens})

                            return out

    def format_token(self, token):
        """
        Converts prodigy token to dict of format:

        {"text":"of","start":32,"end":34,"id":5}
        """
        out = dict()
        out["text"] = token.text
        out["start"] = token.idx
        out["end"] = token.idx + len(token)
        out["id"] = token.i

        return out

    def combine_n_rows(self, doc, n=5, join_char=" "):
        """
        Splits a document into chunks of length `n` lines.

        Args:
            doc(str): A document as a string.
            n(int): The number of lines allowed in each chunk.
            join_char(str): The character used to join lines within a chunk.

        Returns:
            list: A list of chunks containing `n` lines.
        """

        indices = list(range(len(doc)))

        # Split the document into blocks

        groups = list(zip(indices[0::n], indices[n::n]))

        # Iterate through each group of n rows, convert all the items
        # to str, and concatenate into a single string

        out = [join_char.join([str(j) for j in doc[beg:end]]) for beg, end in groups]

        # Check whether there is a remainder and concatenate if so

        max_index = len(groups) * n

        last_group = join_char.join([str(j) for j in doc[max_index:len(doc)]])

        out.append(last_group)

        return out

    def split_lines(self, doc, split_char="\\n", add_linebreak=True):
        """
        Split a document by `split_char`

        Args:
            doc(str): A document containing references
            split_char(str): Character by which `doc` will be split
            add_linebreak(bool): If `True`, re-adds the linebreak character to the
                end of each line that is split.

        Returns:
            (list): List of split lines (str).

        """

        lines = doc.split(split_char)

        if add_linebreak:
            lines = [i + split_char for i in lines]

        return lines



@plac.annotations(
    input_file=(
        "Path to jsonl file containing produced by scraper and containing reference sections.",
        "positional", None, str),
    output_file=(
        "Path to jsonl file into which prodigy format references will be saved.",
        "positional", None, str),
    lines=(
        "How many lines to include in an annotation example.",
        "option", "l", int),
    split_char=("Which character to split lines on.", "option", "s", str),
    no_linebreak=(
        "Don't re-add linebreaks to the annotation examples after splitting.",
        "flag", "n", str),
    join_char=(
        "Which character should be used to join lines into an annotation example.",
        "option", "j", str),
)
def reach_to_prodigy(input_file, output_file, lines=10, split_char="\\n",
    no_linebreak=False, join_char=" "):

    print(split_char)

    scraped_json = read_jsonl(input_file)

    logger.info("Loaded %s scraped examples", len(scraped_json))

    if no_linebreak:
        add_linebreak = False
    else:
        add_linebreak = True

    prodigy_format_references = ReachToProdigy(
        scraped_json, lines=lines, split_char=split_char,
        add_linebreak=add_linebreak, join_char=join_char
    )

    references = prodigy_format_references.run()

    write_jsonl(references, output_file=output_file)

    logger.info("Prodigy format written to %s", output_file)
