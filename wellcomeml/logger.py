#!/usr/bin/env python3
# coding: utf-8

"""
Set up shared logger
"""

import logging
import os


def build_logger(logging_level, name):

    if isinstance(logging_level, str):
        numeric_level = getattr(logging, logging_level.upper(), 10)
    else:
        numeric_level = 20

    logger = logging.getLogger(name)

    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=numeric_level,
    )

    return logger


LOGGING_LEVEL = os.getenv("LOGGING_LEVEL")

logger = build_logger(logging_level=LOGGING_LEVEL, name=__name__)

external_logging_level = {'transformers': logging.WARNING}

for package, level in external_logging_level.items():
    logging.getLogger(package).setLevel(level)
