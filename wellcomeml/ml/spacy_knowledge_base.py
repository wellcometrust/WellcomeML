"""
Creates a knowledge base using the vocab from an NLP model
and pretrains the entity encodings using the entity descriptions

See https://spacy.io/usage/training#entity-linker for where I got this code from
"""
from pathlib import Path
import subprocess
import os

from wellcomeml.utils import throw_extra_import_message

try:
    from spacy.vocab import Vocab
    from spacy.kb import KnowledgeBase

    import spacy
except ImportError as e:
    throw_extra_import_message(error=e, required_module='spacy', extra='spacy')


class SpacyKnowledgeBase(object):
    def __init__(
        self, kb_model="en_core_web_lg", desc_width=64, input_dim=300, num_epochs=5
    ):
        """
        Input:
            kb_model: spacy pretrained model with word embeddings
            desc_width: length of entity vectors
            input_dim: dimension of pretrained input vectors
            num_epochs: number of epochs in training entity encodings
        """
        self.kb_model = kb_model
        self.desc_width = desc_width
        self.input_dim = input_dim
        self.num_epochs = num_epochs

    def train(self, entities, list_aliases):
        """
        Args:
            entities: a dict of each entity, it's description and it's corpus frequency
            list_aliases: a list of dicts for each entity e.g.::

                    [{
                        'alias':'Farrar',
                        'entities': ['Q1', 'Q2'],
                        'probabilities': [0.4, 0.6]
                    }]

                probabilities are 'prior probabilities' and must sum to < 1
        """
        try:
            nlp = spacy.load(self.kb_model)
        except IOError:
            subprocess.run(["python", "-m", "spacy", "download", self.kb_model])
            # pkg_resources need to be reloaded to pick up the newly installed models
            import pkg_resources
            import imp

            imp.reload(pkg_resources)
            nlp = spacy.load(self.kb_model)

        print("Loaded model '%s'" % self.kb_model)

        # set up the data
        entity_ids = []
        embeddings = []
        freqs = []
        for key, value in entities.items():
            desc, freq = value
            entity_ids.append(key)
            embeddings.append(nlp(desc).vector)
            freqs.append(freq)

        self.entity_vector_length = len(embeddings[0])  # This is needed in loading a kb
        kb = KnowledgeBase(
            vocab=nlp.vocab, entity_vector_length=self.entity_vector_length
        )

        # set the entities, can also be done by calling `kb.add_entity` for each entity
        kb.set_entities(entity_list=entity_ids, freq_list=freqs, vector_list=embeddings)

        # adding aliases, the entities need to be defined in the KB beforehand
        for alias in list_aliases:
            kb.add_alias(
                alias=alias["alias"],
                entities=alias["entities"],
                probabilities=alias["probabilities"],
            )
        self.kb = kb
        return self.kb

    def save(self, output_dir):
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        kb_path = os.path.join(output_dir, "kb")
        self.kb.to_disk(kb_path)
        print("Saved KB to", kb_path)

        vocab_path = os.path.join(output_dir, "vocab")
        self.kb.vocab.to_disk(vocab_path)
        print("Saved vocab to", vocab_path)

        kb_info_path = os.path.join(output_dir, "kb_info.txt")
        with open(kb_info_path, "w") as file:
            # The first line must be the entity_vector_length for load to work
            file.write(f"{self.entity_vector_length} \n")
            file.write(f"{self.kb_model} \n")
        print("Saved knowledge base info to", kb_info_path)

    def load(self, output_dir):
        kb_path = os.path.join(output_dir, "kb")
        vocab_path = os.path.join(output_dir, "vocab")
        kb_info_path = os.path.join(output_dir, "kb_info.txt")
        print("Loading vocab from", vocab_path)
        print("Loading KB from", kb_path)
        print("Loading KB info from", kb_info_path)
        with open(kb_info_path, "r") as file:
            # The first line is the entity_vector_length
            entity_vector_length = int(file.readline().strip())
        vocab = Vocab().from_disk(vocab_path)
        kb = KnowledgeBase(vocab=vocab, entity_vector_length=entity_vector_length)
        kb.from_disk(kb_path)
        self.kb = kb
        return self.kb

    def __str__(self):
        print(self.kb.get_size_entities(), "kb entities:", self.kb.get_entity_strings())
        print(self.kb.get_size_aliases(), "kb aliases:", self.kb.get_alias_strings())
