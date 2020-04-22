import random
import csv
import os

from wellcomeml.datasets.download import check_cache_and_download

def load_split(data_path):
    X = []
    Y = []
    with open(data_path) as f:
        csvreader = csv.DictReader(f, delimiter='\t')
        for line in csvreader:
            X.append(line["sentence"])
            Y.append(line["labels"].split(','))
    return X, Y

def load_hoc(split='train', shuffle=True):
    path = check_cache_and_download("hoc")

    train_data_path = os.path.join(path, "train.tsv")
    X_train, Y_train = load_split(train_data_path)

    test_data_path = os.path.join(path, "test.tsv")
    X_test, Y_test = load_split(test_data_path)

    if split == 'train':
        X = X_train
        Y = Y_train
    elif split == 'test':
        X = X_test
        Y = Y_test
    else:
        X = X_train + X_test
        Y = Y_train + Y_test

    if shuffle:
        data = list(zip(X, Y))
        shuffled_data = random.shuffle(data)
        X, Y = zip(*data)

    return X, Y
