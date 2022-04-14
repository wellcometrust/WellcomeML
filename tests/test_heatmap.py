import os

from wellcomeml.viz.elements import plot_heatmap


def test_heatmap(tmp_path):

    co_occurrence = [
       {"concept_1": "Data Science", "concept_2": "Machine Learning", "value": 1, "abbr": "DS/ML"},
       {"concept_1": "Machine Learning", "concept_2": "Data Science", "value": 0.3, "abbr": "ML/DS"},
       {"concept_1": "Science", "concept_2": "Data Science", "value": 1, "abbr": "S/DS"},
       {"concept_1": "Science", "concept_2": "Machine Learning", "value": 1, "abbr": "S/ML"},
       {"concept_1": "Data Science", "concept_2": "Science", "value": 0.05, "abbr": "DS/S"},
       {"concept_1": "Machine Learning", "concept_2": "Science", "value": 0.01, "abbr": "ML/S"}
    ]

    # Plot in blue
    path = os.path.join(tmp_path, 'test.html')

    plot_heatmap(co_occurrence, file=path, color="Blue Lagoon",
                 metadata_to_display=[("Abbreviation", "abbr")])

    # Asserts that it created the file correctly.
    assert os.path.exists(path)

