"""
A class that for each of a list of sentences will find the most similar document in a corpus
using the TFIDF vectors or a BERT embedding from the corpus documents.

"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from wellcomeml.ml import BertVectorizer


class SimilarityEntityLinker:
    def __init__(self, stopwords, embedding="tf-idf"):
        """
        Input:
            stopwords - list of stopwords
            embedding - How to embed the documents
                    in order to find which document in the corpus
                    is most similar to the sentence.
                    embedding='tf-idf': Use a TFIDF vectoriser
                    embedding='bert': Use a BERT vectoriser
        """

        self.stopwords = stopwords
        self.embedding = embedding

    def _clean_text(self, text):
        """
        Clean a body of text
        """
        text_split = text.replace("\n", " ")

        return text_split

    def _clean_kb(self, raw_knowledge_base):
        """
        Creates a cleaned version of the raw_knowledge_base
        which is a dictionary of each document's text.

        Don't include any empty text information
        """

        knowledge_base = {}
        for key, text in raw_knowledge_base.items():
            if len(text.replace(" ", "")) != 0:
                knowledge_base[key] = self._clean_text(text)

        return knowledge_base

    def fit(self, documents):
        """
        documents: dictionary of the texts from each of the corpus documents
        """

        documents = self._clean_kb(documents)

        document_texts = list(documents.values())
        self.classifications = list(documents.keys())

        if self.embedding == "tf-idf":
            self.vectorizer = TfidfVectorizer(stop_words=self.stopwords)
            self.corpus_matrix = self.vectorizer.fit_transform(document_texts)
        else:
            self.vectorizer = BertVectorizer(sentence_embedding="mean_last")
            self.vectorizer.fit()
            self.corpus_matrix = self.vectorizer.transform(document_texts)

    def predict_proba(self, data):
        """
        Returns probability estimates for each class
        in the same order as self.classifications
        """

        sentences = [self._clean_text(sentence) for sentence, _ in data]
        query = self.vectorizer.transform(sentences)

        class_probabilities = cosine_similarity(query, self.corpus_matrix)

        return class_probabilities

    def optimise_threshold(self, data, id_col="orcid", no_id_col="No ORCID"):
        """
        Find the f1 scores when different similarity thresholds are used.
        Use half the maximum F1 score found as the optimal threshold.
        This value will be used as the value for the similarity
        threshold in predict, unless another value is given.
        """

        y_true = [test_meta[id_col] for _, test_meta in data]

        # Find the best prediction and probability for each data point
        probabilities = self.predict_proba(data)
        pred_entities = []
        pred_similarities = []
        for entity, similarity in zip(
            np.argmax(probabilities, axis=1), np.max(probabilities, axis=1)
        ):
            pred_entities.append(self.classifications[entity])
            pred_similarities.append(similarity)

        # Lipton, Z. C., Elkan, C., & Naryanaswamy, B. (2014)
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4442797/
        f1_scores = []
        for similarity_threshold in np.linspace(0, 1, 40):
            pred_entities_temp = [
                pred_entities[i] if sim <= similarity_threshold else no_id_col
                for i, sim in enumerate(pred_similarities)
            ]
            f1_scores.append(
                f1_score(
                    y_true, pred_entities_temp, average="weighted", zero_division=0
                )
            )
        self.optimal_threshold = max(f1_scores) / 2

    def predict(self, data, similarity_threshold=None, no_id_col="No ORCID"):
        """
        Identify the most similar document to a sentence using TFIDF

        If the most similar document doesnt have a similarity value over
        a threshold then return they key 'No ORCID'

        similarity_threshold can be specified, otherwise if you've optimised
        the threshold it will use this value

        Input:
            data: a list of tuples in the form
                [('A sentence about Farrar',
                {metadata}),
                ('A sentence about Smith',
                {metadata})]
                For this predict function the contents of
                {metadata} isn't important
            similarity_threshold: The threshold by which to
                classify a match as being true or that there is
                no match. If this is None then the best threshold
                will be found
        Output:
            pred_entities: a list of predictions
                of which document in the corpus each data
                point is likely to link to
                ['0000-0002-2700-623X', '0000-0002-6259-1606', 'No ORCID']
        """

        if similarity_threshold is None:
            similarity_threshold = self.optimal_threshold

        # Find all the probabilities for the different classes
        probabilities = self.predict_proba(data)

        pred_entities = []
        for entity, similarity in zip(
            np.argmax(probabilities, axis=1), np.max(probabilities, axis=1)
        ):
            if similarity > similarity_threshold:
                pred_entities.append(self.classifications[entity])
            else:
                pred_entities.append(no_id_col)

        return pred_entities

    def evaluate(self, y_true, y_pred):

        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

        return f1_micro
