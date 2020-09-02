Example usage
======================================
Below are some examples of modules.

Bert and Scibert Vectorisers
----------------------------

This vectoriser uses excellent `hugging face transformers <https://github.com/huggingface/transformers>`_ embeddings.
In order to transform text:

.. code-block:: python

    from wellcomeml.ml import BertVectorizer


    X = [
        "Malaria is a disease that kills people",
        "Heart problems comes first in the global burden of disease",
        "Wellcome also funds policy and culture research"
    ]

    vectorizer = BertVectorizer()
    X_transformed = vectorizer.fit_transform(X)

    print(cosine_similarity(X_transformed))

The :class:`BertVectorizer <wellcomeml.ml.bert_vectorizer>` admits an initialisation parameter `pretrained`, which
can be switched to `scibert`, maintaining the code structure but switching the enmbedding

Bert and Scibert Classifiers
----------------------------
The same way as the bert vectorisers, one can use a wrapper to train a text classifier using bert or scibert as base,
using a :class:`BertClassifier <wellcomeml.ml.bert_classifier>`

.. code-block:: python

    import numpy as np

    from wellcomeml.ml import BertClassifier

    X = ["Hot and cold", "Hot", "Cold"]
    Y = np.array([[1,1],[1,0],[0,1]])

    bert = BertClassifier()
    bert.fit(X, Y)
    print(bert.score(X, Y))


Train a Spacy NER model
----------------------------
Retrain a `spaCy NER classifier <https://spacy.io/usage/training#ner>` on new data using :class:`SpacyNER <wellcomeml.ml.spacy_ner>`.

.. code-block:: python

    import random

    from wellcomeml.ml import SpacyNER

    X_train = [
        "n Journal of Psychiatry 158: 2071–4\nFreeman MP, Hibbeln JR, Wisner KL et al. (2006)\n",
        "rd, (BKKBN)\n \nJakarta, Indonesia\n29. Drs Titut Prihyugiarto\n MSPA\n \n",
        "a Santé, 2008. \n118. Konradsen, F. et coll. Community uptake of safe ",
    ]
    y_train = [
        [
            {"start": 36, "end": 46, "label": "PERSON"},
            {"start": 48, "end": 58, "label": "PERSON"},
            {"start": 61, "end": 69, "label": "PERSON"},
        ],
        [
          {"start": 41, "end": 59, "label": "PERSON"}
        ],
        [
          {"start": 21, "end": 34, "label": "PERSON"}
        ],
    ]
    person_tag_name = "PERSON"

    spacy_ner = SpacyNER(n_iter=3, dropout=0.2, output=True)
    spacy_ner.load("en_core_web_sm")
    nlp = spacy_ner.fit(X_train, y_train)

    # Predict the entities in a piece of text
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
    print(spacy_ner.score(y_train, y_pred, tags=[person_tag_name]))
