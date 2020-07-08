import os
import tarfile

import boto3
from botocore import UNSIGNED
from botocore.client import Config

from wellcomeml.logger import logger

DATA_DIR = os.path.expanduser("~/.cache/wellcomeml/data")
DATA_DISPATCH = {
    "hoc": {
        "bucket": "datalabs-public",
        "path": "datasets/hoc/hoc.tar",
        "file_name": "hoc.tar",
    },
    "winer": {
        "bucket": "datalabs-public",
        "path": "datasets/ner/winer.tar",
        "file_name": "winer.tar",
    },
    "conll": {
        "bucket": "datalabs-public",
        "path": "datasets/ner/conll.tar",
        "file_name": "conll.tar",
    },
}


def check_cache_and_download(dataset_name):
    """ Checks if dataset_name is cached and return complete path"""
    os.makedirs(DATA_DIR, exist_ok=True)

    dataset_path = os.path.join(DATA_DIR, dataset_name)
    if not os.path.exists(dataset_path):
        logger.info(f"Could not find dataset {dataset_name}. Downloading from S3")

        # The following allows to download from S3 without AWS credentials
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        tmp_file = os.path.join(DATA_DIR, DATA_DISPATCH[dataset_name]["file_name"])

        s3.download_file(
            DATA_DISPATCH[dataset_name]["bucket"],
            DATA_DISPATCH[dataset_name]["path"],
            tmp_file,
        )

        tar = tarfile.open(tmp_file)
        tar.extractall(path=DATA_DIR)
        tar.close()

        os.remove(tmp_file)

    return dataset_path
