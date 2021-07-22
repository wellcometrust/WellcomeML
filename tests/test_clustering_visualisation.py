import pytest
import os

from wellcomeml.ml.clustering import TextClustering
from wellcomeml.viz.visualize_clusters import visualize_clusters


def test_output_html(tmp_path):
    """Tests that the output html is generated correclty by the clustering function"""

    # This will be the file to
    temporary_file = os.path.join(tmp_path, 'test-cluster.html')

    # Run clustering on small dummy data (see test_clustering.py)
    cluster = TextClustering(embedding_random_state=42,
                             reducer_random_state=43,
                             clustering_random_state=44)

    X = ['Wellcome Trust',
         'The Wellcome Trust',
         'Sir Henry Wellcome',
         'Francis Crick',
         'Crick Institute',
         'Francis Harry Crick']

    cluster.fit(X)

    # Run the visualisation function with output_file=temporary_file
    visualize_clusters(clustering=cluster, output_file_path=temporary_file, radius=0.01,
                       alpha=0.5, output_in_notebook=False)

    # Assert that the html was generated correctly
    assert os.path.exists(temporary_file)
