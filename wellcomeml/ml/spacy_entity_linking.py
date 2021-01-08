"""
TODO: Fill this
"""
from pathlib import Path
import random

from spacy.util import minibatch, compounding
import spacy

from wellcomeml.ml.spacy_knowledge_base import SpacyKnowledgeBase


class SpacyEntityLinker(object):
    def __init__(self, kb_path, n_iter=50, print_output=True):
        self.kb_path = kb_path
        self.n_iter = n_iter
        self.print_output = print_output

    def _remove_examples_not_in_kb(self, data):
        # Remove examples with unknown identifiers to the knowledge base
        kb_ids = self.nlp.get_pipe("entity_linker").kb.get_entity_strings()
        train_docs = []
        for text, annotation in data:
            with self.nlp.disable_pipes("entity_linker"):
                doc = self.nlp(text)
            annotation_clean = annotation
            for offset, kb_id_dict in annotation["links"].items():
                new_dict = {}
                for kb_id, value in kb_id_dict.items():
                    if kb_id in kb_ids:
                        new_dict[kb_id] = value
                    else:
                        print(
                            "Removed",
                            kb_id,
                            "from training because it is not in the KB.",
                        )
                annotation_clean["links"][offset] = new_dict
            train_docs.append((doc, annotation_clean))
        return train_docs

    def train(self, data):
        """
        Args:
            data: list of training data in the form::

                    [('A sentence about Farrar',
                    {'links': {(17, 22): {'Q1': 1.0, 'Q2': 0.0}}})]

        See https://spacy.io/usage/linguistic-features#entity-linking
        for where I got this code from
        """
        # TODO: Replace n_iter with self.n_iter
        n_iter = self.n_iter

        kb = SpacyKnowledgeBase()
        kb = kb.load(self.kb_path)
        print("Loaded Knowledge Base from '%s'" % self.kb_path)

        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.vocab.vectors.name = "spacy_pretrained_vectors"

        self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))

        # Create the Entity Linker component and add it to the pipeline.
        if "entity_linker" not in self.nlp.pipe_names:
            entity_linker = self.nlp.create_pipe("entity_linker")
            entity_linker.set_kb(kb)
            self.nlp.add_pipe(entity_linker, last=True)

        data = self._remove_examples_not_in_kb(data)

        pipe_exceptions = ["entity_linker"]
        other_pipes = [
            pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions
        ]
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            for itn in range(n_iter):
                random.shuffle(data)
                losses = {}
                batches = minibatch(data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.nlp.update(
                        texts,
                        annotations,
                        drop=0.2,
                        losses=losses,
                        sgd=optimizer,
                    )
                if self.print_output:
                    print(itn, "Losses", losses)

        return self.nlp

    def _get_token_nums(self, doc, char_idx):
        """
        Convert a character index to a token index
        i.e. what number token is character number char_idx in ?
        """
        for i, token in enumerate(doc):
            if char_idx > token.idx:
                continue
            if char_idx == token.idx:
                return i
            if char_idx < token.idx:
                return i

    def predict(self, data):
        """
        See how well the model predicts which entity you are referring to in your data

        Args:
            data: list of test data in the form::

                    [('A sentence about Farrar',
                    {'links': {(17, 22): {'Q1': 1.0, 'Q2': 0.0}}})]

        Returns:
            list: pred_entities_ids: [['Q1'], ['Q1'], ['Q2']
        """
        pred_entities_ids = []
        for text, annotation in data:
            doc = self.nlp(text)
            names = [text[s:e] for s, e in annotation["links"].keys()]
            doc_entities_ids = []
            for ent in doc.ents:
                if (ent.label_ == "PERSON") and (ent.text in names):
                    doc_entities_ids.append(ent.kb_id_)
            pred_entities_ids.append(doc_entities_ids)

        return pred_entities_ids

    def save(self, output_dir):
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        self.nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

    def load(self, output_dir):
        print("Loading from", output_dir)
        self.nlp = spacy.load(output_dir)
        return self.nlp
