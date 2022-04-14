from wellcomeml.viz.elements import plot_heatmap

# Fake co_occurrence matrix of the concepts "Data Science", "Machine Learning" and "Science"

co_occurrence = [
   {"concept_1": "Data Science", "concept_2": "Machine Learning", "value": 1, "abbr": "DS/ML"},
   {"concept_1": "Machine Learning", "concept_2": "Data Science", "value": 0.3, "abbr": "ML/DS"},
   {"concept_1": "Science", "concept_2": "Data Science", "value": 1, "abbr": "S/DS"},
   {"concept_1": "Science", "concept_2": "Machine Learning", "value": 1, "abbr": "S/ML"},
   {"concept_1": "Data Science", "concept_2": "Science", "value": 0.05, "abbr": "DS/S"},
   {"concept_1": "Machine Learning", "concept_2": "Science", "value": 0.01, "abbr": "ML/S"}
]

# Plot in blue
plot_heatmap(co_occurrence, file='test-blue.html', color="Blue Lagoon",
             metadata_to_display=[("Abbreviation", "abbr")])

# Plot in gold
plot_heatmap(co_occurrence, file='test-gold.html', color="Tahiti Gold",
             metadata_to_display=[("Abbreviation", "abbr")])
