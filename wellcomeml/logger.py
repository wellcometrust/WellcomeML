#!/usr/bin/env python3
# coding: utf-8

"""
Set up shared logger
"""
import logging
import warnings
import os


def get_numeric_level(level):
    if isinstance(level, str):
        level = getattr(logging, level.upper(), 10)
    return level


def build_logger(logging_level, name):
    numeric_level = get_numeric_level(logging_level)

    logger = logging.getLogger(name)

    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=numeric_level,
    )

    return logger


DEFAULT_LOGGING_LEVEL = "INFO"
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", DEFAULT_LOGGING_LEVEL)
NUMERIC_LEVEL = get_numeric_level(LOGGING_LEVEL)

logger = build_logger(logging_level=NUMERIC_LEVEL, name=__name__)

external_logging_level = {
    'transformers': NUMERIC_LEVEL,
    'tensorflow': NUMERIC_LEVEL,
    'gensim': NUMERIC_LEVEL,
    'sklearn': NUMERIC_LEVEL,
    'spacy': NUMERIC_LEVEL,
    'torch': NUMERIC_LEVEL,
    'tokenizers': NUMERIC_LEVEL
}

for package, level in external_logging_level.items():
    logging.getLogger(package).setLevel(NUMERIC_LEVEL)

if NUMERIC_LEVEL >= 40:  # ERROR
    warnings.filterwarnings("ignore")
