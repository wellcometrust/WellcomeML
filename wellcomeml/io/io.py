#!/usr/bin/env python3
# coding: utf-8

"""
Utilities for loading and saving data from various formats
"""
import logging
import json

logger = logging.getLogger(__name__)


def write_jsonl(input_data, output_file):
    """
    Write a dict to jsonl (line delimited json)

    Output format will look like:

    ```
    {'a': 0}
    {'b': 1}
    {'c': 2}
    {'d': 3}
    ```

    Args:
        input_data(dict): A dict to be written to json.
        output_file(str): Filename to which the jsonl will be saved.
    """

    with open(output_file, 'w') as fb:

        # Check if a dict (and convert to list if so)

        if isinstance(input_data, dict):
            input_data = [value for key, value in input_data.items()]

        # Write out to jsonl file

        logger.debug('Writing %s lines to %s', len(input_data), output_file)

        for i in input_data:
            json_ = json.dumps(i) + '\n'
            fb.write(json_)


def _yield_jsonl(file_name):
    for row in open(file_name, "r"):
        yield json.loads(row)


def read_jsonl(input_file):
    """Create a list from a jsonl file

    Args:
        input_file(str): File to be loaded.
    """

    out = list(_yield_jsonl(input_file))

    logger.debug('Read %s lines from %s', len(out), input_file)

    return out
