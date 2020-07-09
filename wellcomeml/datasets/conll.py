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


def load_conll(split="train", shuffle=True, inc_outside=True):
    path = check_cache_and_download("conll")

    if split == "train":
        train_data_path = os.path.join(path, "eng.train")
        X, Y = _load_data_spacy(train_data_path, inc_outside=inc_outside)
    elif split == "test":
        test_data_path = os.path.join(path, "eng.testa")
        X, Y = _load_data_spacy(test_data_path, inc_outside=inc_outside)
    elif split == "evaluate":
        eval_data_path = os.path.join(path, "eng.testb")
        X, Y = _load_data_spacy(eval_data_path, inc_outside=inc_outside)
    else:
        raise ValueError(
            f"Split argument {split} is not one of train, test or evaluate"
        )

    if shuffle:
        data = list(zip(X, Y))
        random.shuffle(data)
        X, Y = zip(*data)

    return X, Y
