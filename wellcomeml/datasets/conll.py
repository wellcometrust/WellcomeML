import os
import random

from wellcomeml.datasets.download import check_cache_and_download


def _load_data_spacy(data_path, inc_outside=True):
    """
    Load data in Spacy format:
    X = list of sentences (plural) / documents ['the cat ...', 'some dog...', ...]
    Y = list of list of entity tags for each sentence
        [[{'start': 36, 'end': 46, 'label': 'PERSON'}, {..}, ..], ... ]
    inc_outside = False: don't include none-entities in the output

    Raw format is:
    '-DOCSTART- -X- O O\n\nEU NNP I-NP I-ORG\nrejects VBZ I-VP O\nGerman JJ I-NP I-MISC...'
    where each article is separated by '-DOCSTART- -X- O O\n',
    each sentence is separate by a blank line,
    and the entity information is in the form
    'EU NNP I-NP I-ORG' (A word, a part-of-speech (POS) tag,
        a syntactic chunk tag and the named entity tag)
    """

    X = []
    Y = []
    with open(data_path) as f:
        articles = f.read().split("-DOCSTART- -X- O O\n\n")
        articles = articles[1:]  # The first will be blank

        for article in articles:
            # Each sentence in the article is separated by a blank line
            sentences = article.split("\n\n")

            for sentence in sentences:
                char_i = 0  # A counter for the entity start and end character indices
                sentence_text = ""
                sentence_tags = []
                entities = sentence.split("\n")

                for entity in entities:
                    # Due to the splitting on '\n' sometimes we are left with empty elements

                    if len(entity) != 0:
                        token, _, _, tag = entity.split(" ")
                        sentence_text += token + " "

                        if tag != "O" or inc_outside:
                            sentence_tags.append(
                                {
                                    "start": char_i,
                                    "end": char_i + len(token),
                                    "label": tag,
                                }
                            )
                        char_i += len(token) + 1  # plus 1 for the space separating

                if sentence_tags != []:
                    X.append(sentence_text)
                    Y.append(sentence_tags)

    return X, Y


def load_conll(split="train", shuffle=True, inc_outside=True, dataset: str = "conll"):
    """Load the conll dataset

    Args:
        split(str): Which split of the data to collect, one of ["train", "test",
            "evaluate"].
        shuffle(bool): Should the data be shuffled with random.shuffle?
        inc_outside(bool): Should outside charavters be included?
        dataset(str): Which dataset to load. This defaults to "conll" and should
            only be altered for test purposes in which case it should be set to
            "test_conll".
    """
    path = check_cache_and_download(dataset)

    map = {"train": "eng.train", "test": "eng.testa", "evaluate": "eng.testb"}

    try:
        data_path = os.path.join(path, map[split])
        X, Y = _load_data_spacy(data_path, inc_outside=inc_outside)
    except KeyError:
        raise KeyError(f"Split argument {split} is not one of train, test or evaluate")

    if shuffle:
        data = list(zip(X, Y))
        random.shuffle(data)
        X, Y = zip(*data)

    return X, Y
