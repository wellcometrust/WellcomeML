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
    from .prodigy.reference_to_token_annotations import reference_to_token_annotations

    commands = {
        "refs_to_tokens": reference_to_token_annotations,
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
