#!/usr/bin/env python3
# coding: utf-8

"""
Set up shared logger
"""

import logging
import os

LOGGING_LEVEL = os.getenv("LOGGING_LEVEL")

if isinstance(LOGGING_LEVEL, str):
    numeric_level = getattr(logging, LOGGING_LEVEL.upper(), 10)
else:
    numeric_level = 20

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=numeric_level,
)
