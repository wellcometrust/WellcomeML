from wellcomeml.ml import TextClustering


def test_full_pipeline():
    cluster = TextClustering()

    X = ['Wellcome Trust',
         'The Wellcome Trust',
         'Sir Henry Wellcome',
         'Francis Crick',
         'Crick Institute',
         'Francis Harry Crick']

    cluster.fit(X)

    assert len(cluster.cluster_kws) == len(cluster.cluster_ids) == 6


def test_parameter_search():
    cluster = TextClustering()
    X = ['Wellcome Trust',
         'The Wellcome Trust',
         'Sir Henry Wellcome',
         'Francis Crick',
         'Crick Institute',
         'Francis Harry Crick']

    param_grid = {
        'reducer': {'min_dist': [0.0, 0.2],
                    'n_neighbors': [2, 3, 5],
                    'metric': ['cosine', 'euclidean']},
        'clustering': {'min_samples': [2, 5],
                       'eps': [0.5, 1, 1.5]}
    }

    best_params = cluster.optimise(X, param_grid=param_grid, verbose=1)

    assert len(best_params['params_list']) == 72
    # Asserts that silhouette is at least positive
    assert best_params['silhouette'] > 0
