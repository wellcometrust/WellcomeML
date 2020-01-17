import os.path


def get_path(p):
    return os.path.join(
        os.path.dirname(__file__),
        p
    )

TEST_JSONL = get_path('test_data/test_jsonl.jsonl')
