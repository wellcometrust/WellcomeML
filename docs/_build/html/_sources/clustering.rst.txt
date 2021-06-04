.. _clustering:

Clustering data with WellcomeML
======================================
We have a module for clustering text data with custom transformations. The basic class is
``TextClustering``, which you can import from :py:mod:`wellcomeml.ml.clustering`.

The pipeline usually consists of:

* A vectorizer (for example :py:class:`wellcomeml.ml.frequency_vectorizer.WellcomeTfidf`)
* A dimensionality reduction algorithm (usually ``umap``)
* A clustering algorithm (usually `DBScan <https://scikit-learn.org/stable/modules/clustering.html>`_, but virtually any of sklearn's algorithms will work).

You will initialise the clustering class by invoking:

.. code-block:: python

    from wellcomeml.ml.clustering import TextClustering
    cluster = TextClustering(
        embedding='tf-idf', # Or bert
        reducer='umap', # Or tsne
        clustering='dbscan' # Or kmeans, optics, hdbscan
    )

If you want to change the basic parameters you can pass an additional argument `params`, that receives 'embedding', 'reducer' and 'clustering' as keys. For example,
for changing the DBScan bandwitdh parameter to `eps = 3`, umap's number of neighbors to 3 and to use Scibert pre-trained model you can


.. code-block:: python

    params = {
        'embedding': {'pretrained': 'scibert'},
        'reducer': {'n_neighbors': 3},
        'clustering': {'eps': 3}
    }

    cluster = TextClustering(
        embedding='bert' # Or bert
        reducer='umap',
        clustering='dbscan',
        params=params
    )

There are a couple of ways to fit a model. You can just use the `.fit()` method, which will fit the whole pipeline, as above, or fit intermediate steps, e.g.:

.. code-block:: python


    cluster = TextClustering()

    X = ['Wellcome Trust',
         'The Wellcome Trust',
         'Sir Henry Wellcome',
         'Francis Crick',
         'Crick Institute',
         'Francis Harry Crick']

    cluster.fit_step(X, step='vectorizer')
    cluster.fit_step(step='reducer')
    cluster.fit_step(step='clustering')

    print(f"The shape of the reduced points is {cluster.reduced_points.shape})

This is particularly useful if you want to access intermediate steps of the pipeline. Note how the subsequent steps after the vectorizer don't need `X` to be passed.
This is because the class stores them. Another *very cool* thing you can do if the reducer is uMAP is to pass a list of classes, y, so you will do **supervised** (or **semi-supervised**)
dimensonality reduction. Check the `uMAP docs <https://umap-learn.readthedocs.io/en/latest/supervised.html>`_ for more info. The usage is ``cluster.fit_step(y=[1,1,1,0,0,0], step='reducer')``.

The third way of fitting a model is to use the optimiser. This function leverages sklearns grid search with a custom metric (silhouette score).
Here is a full example for text clustering optimisation:

.. code-block:: python

    from wellcomeml.ml.clustering import TextClustering

    cluster = TextClustering()

    X = ['Wellcome Trust',
         'The Wellcome Trust',
         'Sir Henry Wellcome',
         'Francis Crick',
         'Crick Institute',
         'Francis Harry Crick']


    # This is a better one. Let's optimise for silhouette

    param_grid = {
     'reducer': {'min_dist': [0.0, 0.2],
                 'n_neighbors': [2, 3, 5],
                 'metric': ['cosine', 'euclidean']},
     'clustering': {'min_samples': [2, 5],
                    'eps': [0.5, 1, 1.5]}
    }

    best_params = cluster.optimise(X, param_grid=param_grid, verbose=1)

    print("Awesome clusters:")
    print([(x, cluster) for x, cluster in zip(X, cluster.cluster_ids)])
    print("Keywords:")
    print(cluster.cluster_kws)

After clustering, you can save or load models using ``cluster.save()`` and ``cluster.load()``.
