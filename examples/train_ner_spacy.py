"""
Trains the SpaCy NER model with tagged NER data in SPacy format
returning some predictions and an evaluation of the trained model
"""

import random

from wellcomeml.ml import SpacyNER
from wellcomeml.metrics.ner_classification_report import ner_classification_report
from wellcomeml.datasets.conll import load_conll
from wellcomeml.datasets.winer import load_winer


def custom_data():
    X_train = [
        "n Journal of Psychiatry 158: 2071–4\nFreeman MP, Hibbeln JR, Wisner KL et al. (2006)\n",
        "rd, (BKKBN)\n \nJakarta, Indonesia\n29. Drs Titut Prihyugiarto\n MSPA\n \n",
        "a Santé, 2008. \n118. Konradsen, F. et coll. Community uptake of safe ",
        "ted that the two treatments can \nbe combined. Contrarily, Wapf et al. \n",
        "Int J Tuberc Lung Dis. 2015;19(6):657–62. \n160. Dudley L, Aze",
        ": cohort study. BMJ, 1997, 315:722–729. \nUmesawa M, Iso H, Date C et ",
        "T.A., G. Marland, and R.J. Andres (2010). Global, Regional, and National ",
        "Ian Gr\nMr Ian Graayy\nPrincipal Policy Officer (Public Health and Health ",
        ". \n3. \nFischer G and Stöver H. Assessing the current state of ",
        "ated by\nLlorca et al. (2014) or Pae et al. (2015), or when vortioxetine ",
    ]
    y_train = [
        [
            {"start": 36, "end": 46, "label": "PERSON"},
            {"start": 48, "end": 58, "label": "PERSON"},
            {"start": 61, "end": 69, "label": "PERSON"},
        ],
        [{"start": 41, "end": 59, "label": "PERSON"}],
        [{"start": 21, "end": 34, "label": "PERSON"}],
        [{"start": 58, "end": 62, "label": "PERSON"}],
        [{"start": 48, "end": 56, "label": "PERSON"}],
        [
            {"start": 41, "end": 50, "label": "PERSON"},
            {"start": 52, "end": 57, "label": "PERSON"},
            {"start": 59, "end": 65, "label": "PERSON"},
        ],
        [
            {"start": 6, "end": 16, "label": "PERSON"},
            {"start": 22, "end": 33, "label": "PERSON"},
        ],
        [
            {"start": 0, "end": 6, "label": "PERSON"},
            {"start": 10, "end": 20, "label": "PERSON"},
        ],
        [
            {"start": 7, "end": 16, "label": "PERSON"},
            {"start": 21, "end": 30, "label": "PERSON"},
        ],
        [
            {"start": 8, "end": 14, "label": "PERSON"},
            {"start": 32, "end": 35, "label": "PERSON"},
        ],
    ]
    person_tag_name = "PERSON"
    return X_train, y_train, person_tag_name


for data_type in ["CONLL", "WiNER", "WiNER not merged", "custom"]:
    print("Training spacy NER model with {} dataset".format(data_type))

    if data_type == "CONLL":
        X_train, y_train = load_conll(split="train", shuffle=False, inc_outside=False)
        person_tag_name = "I-PER"
    elif data_type == "WiNER":
        X_train, y_train = load_winer(split="train", shuffle=False, inc_outside=False)
        person_tag_name = "0"
    elif data_type == "WiNER not merged":
        X_train, y_train = load_winer(
            split="train", shuffle=False, inc_outside=False, merge_entities=False
        )
        person_tag_name = "0"
    else:
        X_train, y_train, person_tag_name = custom_data()

    n = 100  # For the purposes of this example just train on a small amount of the data
    X_train = X_train[0:min(n, len(X_train))]
    y_train = y_train[0:min(n, len(y_train))]

    # # A list of the groups each of the data points belong to
    groups = random.choices(["Group 1", "Group 2", "Group 3"], k=len(X_train))

    spacy_ner = SpacyNER(n_iter=3, dropout=0.2, output=True)
    spacy_ner.load("en_core_web_sm")
    nlp = spacy_ner.fit(X_train, y_train)

    # # Predict the entities in a piece of text
    text = (
        "\nKhumalo, Lungile, National Department of Health \n• \nKistnasamy, "
        "Dr Barry, National Department of He"
        )
    predictions = spacy_ner.predict(text)
    print(
        [
            text[entity["start"]:entity["end"]]
            for entity in predictions
            if entity["label"] == person_tag_name
        ]
    )

    # Evaluate the performance of the model on the training data
    y_pred = [spacy_ner.predict(text) for text in X_train]
    f1 = spacy_ner.score(y_train, y_pred, tags=[person_tag_name])
    print(f1)

    # Evaluate the performance of the model per group
    report = ner_classification_report(y_train, y_pred, groups, tags=[person_tag_name])
    print(report)
