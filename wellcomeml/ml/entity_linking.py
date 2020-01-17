"""
TODO: Fill this
"""
from pathlib import Path
import random
import os

from spacy.vocab import Vocab
from spacy.kb import KnowledgeBase
from spacy.symbols import PERSON
from spacy.tokens import Span
from spacy.util import minibatch, compounding
import spacy

from wellcomeml.ml.knowledge_base import PeopleKB


class EntityLinker(object):
    def __init__(self, kb_path, n_iter=50, print_output=True):
        self.kb_path = kb_path
        self.n_iter = n_iter
        self.print_output = print_output

    def _remove_examples_not_in_kb(self, kb, data):
        # Remove examples with unknown identifiers to the knowlege base
        kb_ids = kb.get_entity_strings()
        for text, annotation in data:
            for offset, kb_id_dict in annotation["links"].items():
                new_dict = {}
                for kb_id, value in kb_id_dict.items():
                    if kb_id in kb_ids:
                        new_dict[kb_id] = value
                    else:
                        print(
                            "Removed", kb_id,
                            "from training because it is not in the KB."
                        )
                annotation["links"][offset] = new_dict
        return data

    def train(self, data):
        """
        Input:
            data: list of training data in the form
                [('A sentence about Farrar',
                {'links': {(17, 22): {'Q1': 1.0, 'Q2': 0.0}}})]

        See https://spacy.io/usage/linguistic-features#entity-linking for where I got this code from
        """
        # TODO: Replace n_iter with self.n_iter
        n_iter = self.n_iter

        kb = PeopleKB()
        kb = kb.load(self.kb_path)
        print("Loaded Knowledge Base from '%s'" % self.kb_path)

        nlp = spacy.blank("en", vocab=kb.vocab)
        nlp.vocab.vectors.name = "spacy_pretrained_vectors"
    
        entity_linker = nlp.create_pipe("entity_linker")
        entity_linker.set_kb(kb)
        nlp.add_pipe(entity_linker, last=True)        

        data = self._remove_examples_not_in_kb(kb, data)
        
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "entity_linker"]
        with nlp.disable_pipes(*other_pipes):
            optimizer = nlp.begin_training()
            for itn in range(n_iter):
                random.shuffle(data)
                losses = {}
                batches = minibatch(data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(
                        texts,
                        annotations,
                        drop=0.2,
                        losses=losses,
                        sgd=optimizer,
                    )
                if self.print_output:
                    print(itn, "Losses", losses)

        self.nlp = nlp
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
        Input:
            data: list of test data in the form
                [('A sentence about Farrar',
                {'links': {(17, 22): {'Q1': 1.0, 'Q2': 0.0}}})]
        Output:
           pred_entities_ids: ['Q1', 'Q1', 'Q2']
    """
        # TODO: Replace nlp_el with self.nlp 
        nlp_el = self.nlp

        # IMPORTANT
        #
        # This code assumes each training example has one entity
        # e.g. (start, end) = list(values['links'].keys())[0]
        #
        # If the code has more than one entities we predict the
        # first only  e.g. pred_entity_id = doc.ents[0].kb_id_
        
        pred_entities_ids = []
        for text, values in data:
            (start, end) = list(values['links'].keys())[0]
            entity_text = text[start:end]

            doc = nlp_el.tokenizer(text)
            entity_tokens = nlp_el.tokenizer(entity_text)

            # Set entity span to PERSON
            doc.ents = [
                Span(
                    doc,
                    self._get_token_nums(doc, start),
                    self._get_token_nums(doc, end),
                    label=PERSON
                )
            ]

            doc = nlp_el.get_pipe("entity_linker")(doc)

            pred_entity_id = doc.ents[0].kb_id_
            pred_entities_ids.append(pred_entity_id)

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
