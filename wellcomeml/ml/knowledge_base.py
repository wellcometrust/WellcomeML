"""
Creates a knowledge base using the vocab from an NLP model
and pretrains the entity encodings using the entity descriptions

See https://spacy.io/usage/training#entity-linker for where I got this code from
"""
from pathlib import Path
import os

from spacy.vocab import Vocab
from spacy.kb import KnowledgeBase
# bin is also a spaCy package
from bin.wiki_entity_linking.train_descriptions import EntityEncoder
import spacy

class PeopleKB(object):

    def __init__(self, kb_model="en_core_web_lg", desc_width=64,
                 input_dim=300, num_epochs=5):
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
        entities: a dict of each entity, it's description and it's corpus frequency
        list_aliases: a list of dicts for each entity e.g. 
            [{
                'alias':'Farrar',
                'entities': ['Q1', 'Q2'],
                'probabilities': [0.4, 0.6]
            }]
            probabilities are 'prior probabilities' and must sum to < 1
        """
        nlp = spacy.load(self.kb_model)
        print("Loaded model '%s'" % self.kb_model)
        kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=self.desc_width)

        # set up the data
        entity_ids = []
        descriptions = []
        freqs = []
        for key, value in entities.items():
            desc, freq = value
            entity_ids.append(key)
            descriptions.append(desc)
            freqs.append(freq)
        
        # training entity description encodings
        # this part can easily be replaced with a custom entity encoder
        encoder = EntityEncoder(
            nlp=nlp,
            input_dim=self.input_dim,
            desc_width=self.desc_width,
            epochs =self.num_epochs
        )
    
        encoder.train(description_list=descriptions, to_print=True)
    
        # get the pretrained entity vectors
        embeddings = encoder.apply_encoder(descriptions)

        # set the entities, can also be done by calling `kb.add_entity` for each entity
        kb.set_entities(entity_list=entity_ids, freq_list=freqs, vector_list=embeddings)

        # adding aliases, the entities need to be defined in the KB beforehand
        for alias in list_aliases:
            kb.add_alias(
                alias=alias['alias'],
                entities=alias['entities'],
                probabilities=alias['probabilities'],  
            )
        self.kb = kb
        return self.kb

    def save(self, output_dir):
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        kb_path = os.path.join(output_dir, "kb")
        self.kb.dump(kb_path)
        print("Saved KB to", kb_path)

        vocab_path = os.path.join(output_dir, "vocab")
        self.kb.vocab.to_disk(vocab_path)
        print("Saved vocab to", vocab_path)
    

    def load(self, output_dir):
        kb_path = os.path.join(output_dir, "kb")
        vocab_path = os.path.join(output_dir, "vocab")
        print("Loading vocab from", vocab_path)
        print("Loading KB from", kb_path)
        vocab = Vocab().from_disk(vocab_path)
        kb = KnowledgeBase(vocab=vocab)
        kb.load_bulk(kb_path)
        self.kb = kb
        return self.kb

    def __str__(self):
        print(
            self.kb.get_size_entities(),
            "kb entities:",
            self.kb.get_entity_strings()
        )
        print(
            self.kb.get_size_aliases(),
            "kb aliases:",
            self.kb.get_alias_strings()
        )
