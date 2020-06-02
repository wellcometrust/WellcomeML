from wellcomeml.datasets.download import check_cache_and_download

import os
import random

def load_data_spacy(data_path, inc_outside=True):

    # Load data in Spacy format:
    # X = list of sentences (plural) / documents ['the cat ...', 'some dog...', ...]
    # Y = list of list of entity tags for each sentence [[{'start': 36, 'end': 46, 'label': 'PERSON'}, {..}, ..], ... ]
    # inc_outside = False: don't include none-entities in the output

    X = []
    Y = []
    with open(data_path) as f:
        lines = f.read().split('-DOCSTART- -X- O O\n')
        for line in lines:
            article_text = ''
            char_i = 0
            article_tags = []
            entities = line.split('\n')
            for entity in entities:
                if len(entity) != 0:
                    token, _, _, tag = entity.split(' ')
                    article_text += token + ' '
                    if tag!='O' or inc_outside:
                        article_tags.append({'start': char_i, 'end': char_i+len(token), 'label': tag})
                    char_i += len(token) + 1 # plus 1 for the space separating
            if article_tags!=[]:
                X.append(article_text)
                Y.append(article_tags)

    return X, Y

def load_conll(split='train', shuffle=True, inc_outside=True):
    path = check_cache_and_download("conll")

    if split == 'train':
        train_data_path = os.path.join(path, "eng.train")
        X, Y = load_data_spacy(train_data_path, inc_outside=inc_outside)
    elif split == 'test':
        test_data_path = os.path.join(path, "eng.testa")
        X, Y = load_data_spacy(test_data_path, inc_outside=inc_outside)
    elif split == 'evaluate':
        eval_data_path = os.path.join(path, "eng.testb")
        X, Y = load_data_spacy(eval_data_path, inc_outside=inc_outside)
    else:
        raise ValueError(f"Split argument {split} is not one of train, test or evaluate")

    if shuffle:
        data = list(zip(X, Y))
        shuffled_data = random.shuffle(data)
        X, Y = zip(*shuffled_data)

    return X, Y

