from wellcomeml.ml import SimilarityEntityLinker

entities_kb = {
    "Michelle Williams (actor)": "American actress. She is the recipient of several accolades, including two Golden Globe Awards and a Primetime Emmy Award, in addition to nominations for four Academy Awards and one Tony Award.",
    "Michelle Williams (musician)": "American entertainer. She rose to fame in the 2000s as a member of R&B girl group Destiny's Child, one of the best-selling female groups of all time with over 60 million records, of which more than 35 million copies sold with the trio lineup with Williams.",
    "id_3": "  ",
}

stopwords = ["the", "and", "if", "in", "a"]

train_data = [
    (
        "After Destiny's Child's disbanded in 2006, Michelle Williams released her first pop album, Unexpected (2008),",
        {"id": "Michelle Williams (musician)"},
    ),
    (
        "On Broadway, Michelle Williams starred in revivals of the musical Cabaret in 2014 and the drama Blackbird in 2016, for which she received a nomination for the Tony Award for Best Actress in a Play.",
        {"id": "Michelle Williams (actor)"},
    ),
    (
        "Franklin would have ideally been awarded a Nobel Prize in Chemistry",
        {"id": "No ID"},
    ),
]

entity_linker = SimilarityEntityLinker(stopwords=stopwords, embedding="tf-idf")
entity_linker.fit(entities_kb)
tfidf_predictions = entity_linker.predict(
    train_data, similarity_threshold=0.1, no_id_col="No ID"
)

entity_linker = SimilarityEntityLinker(stopwords=stopwords, embedding="bert")
entity_linker.fit(entities_kb)
bert_predictions = entity_linker.predict(
    train_data, similarity_threshold=0.1, no_id_col="No ID"
)

print("TF-IDF Predictions:")
for i, (sentence, _) in enumerate(train_data):
    print(sentence)
    print(tfidf_predictions[i])

print("BERT Predictions:")
for i, (sentence, _) in enumerate(train_data):
    print(sentence)
    print(bert_predictions[i])
