import logging
import tarfile
import os

import boto3
from botocore import UNSIGNED
from botocore.client import Config

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.expanduser("~/.cache/wellcomeml/models")

MODEL_DISPATCH = {
    'scibert_scivocab_uncased': {
        "bucket": "ai2-s2-research",
        "path": "scibert/huggingface_pytorch/scibert_scivocab_uncased.tar",
        "file_name": "scibert_scivocab_uncased.tar"
    },
    'scibert_scivocab_cased': {
        "bucket": "ai2-s2-research",
        "path": "scibert/huggingface_pytorch/scibert_scivocab_cased.tar",
        "file_name": "scibert_scivocab_cased.tar"
    },
    'biosent2vec': {
        "bucket": "datalabs-public",
        "path": "models/ncbi-nlp/biosent2vec.bin",
        "file_name": "biosent2vec.bin",
    },
    'sent2vec_wiki_unigrams': {
        "bucket": "datalabs-public",
        "path": "models/epfml/wiki_unigrams.bin",
        "file_name": "wiki_unigrams.bin",
    }
}


def check_cache_and_download(model_name):
    """ Checks if model_name is cached and return complete path"""
    os.makedirs(MODELS_DIR, exist_ok=True)

    FILE_NAME = MODEL_DISPATCH[model_name]['file_name']
    _, FILE_EXT = FILE_NAME.split(".")

    model_path = os.path.join(MODELS_DIR, model_name if FILE_EXT == "tar" else FILE_NAME)
    if not os.path.exists(model_path):
        logger.info(f"Could not find model {model_name}. Downloading from S3")

        # The following allows to download from S3 without AWS credentials
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        tmp_file = os.path.join(MODELS_DIR, FILE_NAME)

        s3.download_file(MODEL_DISPATCH[model_name]['bucket'],
                         MODEL_DISPATCH[model_name]['path'], tmp_file)

        if FILE_EXT == 'tar':
            tar = tarfile.open(tmp_file)
            tar.extractall(path=MODELS_DIR)
            tar.close()

            os.remove(tmp_file)

    return model_path


def throw_extra_import_message(error, extra, required_module):
    """Safely throws an import error if it due to missing extras, and re-raising it otherwise"""
    if error.name == required_module:
        raise ImportError(f"To use this class/module you need to install wellcomeml with {extra} "
                          f"extras, e.g. pip install wellcomeml[{extra}]")
    else:
        raise error
