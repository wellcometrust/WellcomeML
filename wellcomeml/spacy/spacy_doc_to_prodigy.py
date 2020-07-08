#!/usr/bin/env python3
# coding: utf-8

import spacy


class SpacyDocToProdigy:
    """Convert spacy documents into prodigy format
    """

    def run(self, docs):
        """
        Cycle through docs and return prodigy docs.
        """

        return list(self.return_one_prodigy_doc(doc) for doc in docs)

    def return_one_prodigy_doc(self, doc):
        """Given one spacy document, yield a prodigy style dict

        Args:
            doc (spacy.tokens.doc.Doc): A spacy document

        Returns:
            dict: Prodigy style document

        """

        if not isinstance(doc, spacy.tokens.doc.Doc):
            raise TypeError("doc must be of type spacy.tokens.doc.Doc")

        text = doc.text
        spans = []
        tokens = []

        for token in doc:
            tokens.append({
                "text": token.text,
                "start": token.idx,
                "end": token.idx + len(token.text),
                "id": token.i,
            })

        for ent in doc.ents:
            spans.append({
                "token_start": ent.start,
                "token_end": ent.end,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
            })

        out = {
            "text": text,
            "spans": spans,
            "tokens": tokens,
        }

        return out
