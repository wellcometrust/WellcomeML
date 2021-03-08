import os
import pandas as pd

from wellcomeml.ml.bert_semantic_equivalence import SemanticEquivalenceClassifier

data_file_path = os.path.join(
    os.path.dirname(__file__), "data/text_similarity_sample_100_pairs.csv"
)

# Reads sample data and formats it
df = pd.read_csv(data_file_path)

X = df[["text_1", "text_2"]].values.tolist()
y = df["label"].values

# Define the classifier and fits for 1 epoch
classifier = SemanticEquivalenceClassifier(
    pretrained="scibert", batch_size=8, eval_batch_size=16
)

classifier.fit(X, y, epochs=1)

test_pair = (
    "the FCC will not request personal identifying information ",
    "personal information will not be requested by the FCC",
)

score_related = classifier.predict_proba([test_pair])

print(f"Sentences are probably related with score {score_related[0][1]}.")
