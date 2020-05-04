import os

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
    }
}
