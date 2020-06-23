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
