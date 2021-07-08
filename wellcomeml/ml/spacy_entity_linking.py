"""
TODO: Fill this
"""
from pathlib import Path
import random

from wellcomeml.utils import throw_extra_import_message

try:
    from spacy.training import Example
    from spacy.util import minibatch, compounding
    import spacy
    from spacy.kb import KnowledgeBase
except ImportError as e:
    throw_extra_import_message(error=e, required_module='spacy', extra='spacy')


class SpacyEntityLinker(object):
    def __init__(self, kb_path, n_iter=50, print_output=True):
        self.kb_path = kb_path
        self.n_iter = n_iter
        self.print_output = print_output

    def _format_examples(self, data):
        # Remove examples with unknown identifiers to the knowledge base
        # Convert text to spacy.tokens.doc.Doc format
        # Return list of Examples objects
        kb_ids = self.nlp.get_pipe("entity_linker").kb.get_entity_strings()
        examples = []
        for text, annotation in data:
            with self.nlp.select_pipes(disable="entity_linker"):
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
            example = Example.from_dict(doc, annotation_clean)
            examples.append(example)
        return examples

    def train(self, data):
        """
        Args:
            data: list of training data in the form::

                    [('A sentence about Farrar',
                    {'links': {(17, 22): {'Q1': 1.0, 'Q2': 0.0}}})]

        See https://spacy.io/usage/linguistic-features#entity-linking
        for where I got this code from
        """
        n_iter = self.n_iter

        vocab_folder = self.kb_path + "/vocab"
        kb_folder = self.kb_path + "/kb"

        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.vocab.from_disk(vocab_folder)
        self.nlp.add_pipe("sentencizer", before="parser")

        def create_kb(vocab):
            entity_vector_length = 300
            kb = KnowledgeBase(vocab=vocab, entity_vector_length=entity_vector_length)
            kb.from_disk(kb_folder)
            return kb

        entity_linker = self.nlp.add_pipe("entity_linker")
        entity_linker.set_kb(create_kb)

        examples = self._format_examples(data)

        optimizer = entity_linker.initialize(
            lambda: iter(examples), nlp=self.nlp, kb_loader=create_kb
        )
        with self.nlp.select_pipes(enable=[]):
            for itn in range(n_iter):
                random.shuffle(examples)
                losses = {}
                batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    self.nlp.update(
                        batch,
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
