import pytest

from wellcomeml.ml.clustering import TextClustering


@pytest.mark.parametrize("reducer,cluster_reduced", [("tsne", True),
                                                     ("umap", True),
                                                     ("umap", False)])
def test_full_pipeline(reducer, cluster_reduced, tmp_path):
    cluster = TextClustering(reducer=reducer, cluster_reduced=cluster_reduced,
                             embedding_random_state=42,
                             reducer_random_state=43,
                             clustering_random_state=44)

    X = ['Wellcome Trust',
         'The Wellcome Trust',
         'Sir Henry Wellcome',
         'Francis Crick',
         'Crick Institute',
         'Francis Harry Crick']

    cluster.fit(X)

    assert len(cluster.cluster_kws) == len(cluster.cluster_ids) == 6

    cluster.save(folder=tmp_path)

    cluster_new = TextClustering()
    cluster_new.load(folder=tmp_path)

    # Asserts all coordinates of the loaded points are equal
    assert (cluster_new.embedded_points != cluster.embedded_points).sum() == 0
    assert (cluster_new.reduced_points != cluster.reduced_points).sum() == 0
    assert cluster_new.reducer_class.__class__ == cluster.reducer_class.__class__
    assert cluster_new.clustering_class.__class__ == cluster.clustering_class.__class__


@pytest.mark.parametrize("reducer", ["tsne", "umap"])
def test_parameter_search(reducer):
    cluster = TextClustering(reducer=reducer)
    X = ['Wellcome Trust',
         'The Wellcome Trust',
         'Sir Henry Wellcome',
         'Francis Crick',
         'Crick Institute',
         'Francis Harry Crick']

    param_grid = {
        'reducer': {'min_dist': [0.0],
                    'n_neighbors': [2],
                    'metric': ['cosine', 'euclidean']},
        'clustering': {'min_samples': [2],
                       'eps': [0.5]}
    }

    best_params = cluster.optimise(X, param_grid=param_grid,
                                   verbose=1,
                                   max_noise=1)

    # Asserts it found a parameter
    assert best_params is not None
    # Asserts the cross-validation results are returned correctly
    assert len(cluster.optimise_results['mean_test_silhouette']) == \
           len(cluster.optimise_results['params'])
    # Asserts that silhouette is at least positive (for umap! - tsne dos
    # not work)
    if reducer != "tsne":
        assert cluster.silhouette > 0
