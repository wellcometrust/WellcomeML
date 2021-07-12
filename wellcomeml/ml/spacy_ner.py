import random

from nervaluate import Evaluator

from wellcomeml.utils import throw_extra_import_message

try:
    from spacy.training import Example
    import spacy
except ImportError as e:
    throw_extra_import_message(error=e, required_module='spacy', extra='spacy')


class SpacyNER:
    """
    Train a spaCy NER classifier on texts annotated using Prodigy
    """

    def __init__(self, n_iter=20, dropout=0.2, output=True):

        self.n_iter = n_iter
        self.dropout = dropout
        self.output = output

    def fit(self, X, y):
        """
        X: a list of sentences,
            e.g. ['Professor Smith said', 'authors Smith and Chen found']
        y: a list of prodigy format spans for each element of X,
            e.g. [{'start': 10, 'end': 15, 'label': 'PERSON'},
                  {'start': 8, 'end': 13, 'label': 'PERSON'},
                  {'start':18, 'end':22, 'label':'PERSON'}]
        """

        train_data = list(zip(X, y))

        if "ner" not in self.nlp_model.pipe_names:
            # If you are training on a blank model
            ner = self.nlp_model.add_pipe("ner")
            self.nlp_model.add_pipe(ner, last=True)
        else:
            ner = self.nlp_model.get_pipe("ner")

        # add labels from data

        for spans in y:
            for span in spans:
                ner.add_label(span["label"])

        examples = []
        for text, spans in train_data:
            annotations = self._spans_to_entities(spans)
            doc = self.nlp_model.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)

        other_pipes = [pipe for pipe in self.nlp_model.pipe_names if pipe != "ner"]
        with self.nlp_model.select_pipes(disable=other_pipes):  # only train NER
            optimizer = self.nlp_model.initialize(lambda: examples)

            for i in range(self.n_iter):
                random.shuffle(train_data)
                losses = {}

                for example in examples:
                    self.nlp_model.update(
                        [example],
                        sgd=optimizer,
                        losses=losses,
                        drop=self.dropout,
                    )

                if self.output:
                    self._print_output(i, losses["ner"])

        return self.nlp_model

    def _print_output(self, batch_i, loss):

        dash = "-" * 15

        if batch_i == 0:
            print(dash)
            print("{:8s}{:13s}".format("BATCH |", "LOSS"))
            print(dash)
        print("{:<8d}{:<10.2f}".format(batch_i, loss))

    def predict(self, text):
        """
        Get entity predictions for a piece of text
        """

        doc = self.nlp_model(text)
        pred_entities = []

        for ent in doc.ents:
            pred_entities.append(
                {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
            )

        return pred_entities

    def score(self, y_true, y_pred, tags):
        """
        Evaluate the model's performance on the data
        for entity types in the 'tags' list, e.g. ['PERSON', 'LOC']
        """

        evaluator = Evaluator(y_true, y_pred, tags=tags)
        results, results_by_tag = evaluator.evaluate()

        score = {tag: results_by_tag[tag]["partial"]["f1"] for tag in tags}
        score["Overall"] = results["partial"]["f1"]
        return score

    def save(self, file_name):

        self.nlp_model.to_disk(file_name)

    def load(self, model=None):
        """
        model:
            The model's location ('models/model')
            or the name of a spaCy model ("en_core_web_sm")
            None (default): Then load a blank spacy english model
        """

        if model:
            self.nlp_model = spacy.load(model)
        else:
            self.nlp_model = spacy.blank("en")

    def _spans_to_entities(self, spans):
        """
        Convert prodigy spans to spacy entities

        from:

        ```
        [
            {'start': 36, 'end': 46, 'label': 'PERSON'},
            {'start': 48, 'end': 58, 'label': 'PERSON'},
            {'start': 61, 'end': 69, 'label': 'PERSON'},
        ]
        ```

        to:

        ```
        {
            'entities': [
                    (36, 46, 'PERSON'),
                    (48, 58, 'PERSON'),
                    (61, 69, 'PERSON'),
                ]
        }
        ```
        """

        entities = []

        for span in spans:
            entities.append((span["start"], span["end"], span["label"]))

        return {"entities": entities}
