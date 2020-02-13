# coding: utf8

"""
Modified from https://github.com/explosion/spaCy/blob/master/spacy/__main__.py

Allows CLI functions defined by plac (or argparse) to be called using the
following syntax:

`python -m wellcomeml <command>`
"""

if __name__ == "__main__":
    import plac
    import sys
    from wasabi import msg
    from .prodigy.numbered_reference_annotator import annotate_numbered_references
    from .prodigy.prodigy_to_tsv import prodigy_to_tsv
    from .prodigy.reach_to_prodigy import reach_to_prodigy
    from .prodigy.reference_to_token_annotations import reference_to_token_annotations

    commands = {
        "annotate_numbered_refs": annotate_numbered_references,
        "prodigy_to_tsv": prodigy_to_tsv,
        "reach_to_prodigy": reach_to_prodigy,
        "refs_to_token_annotations": reference_to_token_annotations,
    }

    if len(sys.argv) == 1:
        msg.info("Available commands", ", ".join(commands), exits=1)
    command = sys.argv.pop(1)
    sys.argv[0] = "wellcomeml %s" % command

    if command in commands:
        plac.call(commands[command], sys.argv[1:])
    else:
        available = "Available: {}".format(", ".join(commands))
        msg.fail("Unknown command: {}".format(command), available, exits=1)
