import random
import csv
import os

from wellcomeml.datasets.download import check_cache_and_download


def load_split(data_path):
    X = []
    Y = []
    with open(data_path) as f:
        csvreader = csv.DictReader(f, delimiter="\t")
        for line in csvreader:
            X.append(line["sentence"])
            Y.append(line["labels"].split(","))
    return X, Y


def load_hoc(split="train", shuffle=True):
    path = check_cache_and_download("hoc")

    if split == "train":
        train_data_path = os.path.join(path, "train.tsv")
        X, Y = load_split(train_data_path)
    elif split == "test":
        test_data_path = os.path.join(path, "test.tsv")
        X, Y = load_split(test_data_path)
    else:
        raise ValueError(f"Split argument {split} is not one of train or test")

    if shuffle:
        data = list(zip(X, Y))
        random.shuffle(data)
        X, Y = zip(*data)

    return X, Y
