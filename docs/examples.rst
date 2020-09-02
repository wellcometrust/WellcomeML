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
Retrain a `spaCy NER classifier <https://spacy.io/usage/training#ner>`_ on new data using :class:`SpacyNER <wellcomeml.ml.spacy_ner>`.

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

Entity Linking
----------------------------
Link sentences to the most similar document in a corpus using :class:`SimilarityEntityLinker <wellcomeml.ml.similarity_entity_linking>`.

.. code-block:: python

    from wellcomeml.ml import SimilarityEntityLinker

    entities_kb = {
        "Michelle Williams (actor)": (
            "American actress. She is the recipient of several accolades, including two Golden Globe"
            " Awards and a Primetime Emmy Award, in addition to nominations for four Academy Awards "
            "and one Tony Award."
            ),
        "Michelle Williams (musician)": (
            "American entertainer. She rose to fame in the 2000s as a member of R&B girl group "
            "Destiny's Child, one of the best-selling female groups of all time with over 60 "
            "million records, of which more than 35 million copies sold with the trio lineup "
            "with Williams."
            ),
        "id_3": "  ",
    }

    stopwords = ["the", "and", "if", "in", "a"]

    train_data = [
        (
            (
                "After Destiny's Child's disbanded in 2006, Michelle Williams released her first "
                "pop album, Unexpected (2008),"
            ),
            {"id": "Michelle Williams (musician)"},
        ),
        (
            (
                "On Broadway, Michelle Williams starred in revivals of the musical Cabaret in 2014 "
                "and the drama Blackbird in 2016, for which she received a nomination for the Tony "
                "Award for Best Actress in a Play."
            ),
            {"id": "Michelle Williams (actor)"},
        ),
        (
            "Franklin would have ideally been awarded a Nobel Prize in Chemistry",
            {"id": "No ID"},
        ),
    ]

    entity_linker = SimilarityEntityLinker(stopwords=stopwords, embedding="tf-idf")
    entity_linker.fit(entities_kb)
    tfidf_predictions = entity_linker.predict(
        train_data, similarity_threshold=0.1, no_id_col="No ID"
    )

    entity_linker = SimilarityEntityLinker(stopwords=stopwords, embedding="bert")
    entity_linker.fit(entities_kb)
    bert_predictions = entity_linker.predict(
        train_data, similarity_threshold=0.1, no_id_col="No ID"
    )

    print("TF-IDF Predictions:")
    for i, (sentence, _) in enumerate(train_data):
        print(sentence)
        print(tfidf_predictions[i])

    print("BERT Predictions:")
    for i, (sentence, _) in enumerate(train_data):
        print(sentence)
        print(bert_predictions[i])
