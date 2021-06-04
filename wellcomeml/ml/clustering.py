from collections import defaultdict
import logging
import os
import pickle

from wellcomeml.ml import vectorizer
from wellcomeml.logger import logger

import numpy as np
from sklearn.base import ClusterMixin
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

try:
    from hdbscan import HDBSCAN
    HDBSCAN_INSTALLED = True
except (ValueError, ModuleNotFoundError):
    HDBSCAN_INSTALLED = False
    logger.warning(
        "If you want to use hdbscan you need to run"
        "pip3 install hdbscan --no-cache-dir --no-binary :all: --no-build-isolation "
        "Read more https://github.com/wellcometrust/WellcomeML/issues/197"
    )


CACHE_DIR = os.path.expanduser("~/.cache/wellcomeml")


class TextClustering(object):
    """
    Basic class for clustering pipelines.

    Attributes:
        vectorizer: The embedding Vectorizer object
        reducer: A dimensionality reduction object
        clustering: A clustering model object
        cluster_ids: Ids of the cluster
        cluster_names: Names of the clusters
        cluster_kws: Keywords for the clusters (only if embedding=tf-idf)
    """

    def __init__(self, embedding='tf-idf', reducer='umap', clustering='dbscan',
                 cluster_reduced=True, n_kw=10, params={},
                 embedding_random_state=None, reducer_random_state=None,
                 clustering_random_state=None):
        """
        Initialises a cluster pipeline

        Args:
            embedding(str): 'tf-idf', 'bert', 'keras' or 'doc2vec'.
            reducer(str): 'umap' or 'tsne'
            clustering(str): 'dbscan', 'kmeans', 'optics', 'hdbscan' or any
            class that is a sklearn.base.ClusterMixin
            cluster_reduced(bool): Whether clustering will be applied after
            reducing the datapoints with reducer, or straight after embedding
            n_kw(int): If n_kw > 0 and embedding='tf-idf', the model
            outputs, besides a cluster id, a list of keywords for each
            cluster.
            params(dict): Dictionary containing extra parameters for
            embedding, reducer or clustering models. Example:
            {'clustering': {'eps': 0.1}, 'embedding': {'ngram_range': (1,2)}}
            embedding_random_state(int): Determines random number generation
            for embedding, if applicable
            reducer_random_state(int): Determines random number generation
            for reducer, if applicable
            clustering_random_state(int): Determines random number generation
            for clustering, if applicable

        """
        self.embedding = embedding
        self.vectorizer = vectorizer.Vectorizer(embedding=embedding,
                                                **params.get('embedding', {}))

        if embedding_random_state and hasattr(self.vectorizer.vectorizer,
                                              'random_state'):
            self.vectorizer.vectorizer.random_state = embedding_random_state

        self.n_kw = n_kw
        self.cluster_reduced = cluster_reduced

        reducer_dispatcher = {
            'tsne': TSNE
        }
        if reducer == "umap":
            # this imports takes 10-20s https://github.com/lmcinnes/umap/issues/631
            import umap
            reducer_dispatcher["umap"] = umap.UMAP
        self.reducer = reducer
        self.reducer_class = reducer_dispatcher[reducer](
            **params.get('reducer', {})
        )

        if reducer_random_state and hasattr(self.reducer_class,
                                            'random_state'):
            self.reducer_class.random_state = reducer_random_state

        clustering_dispatcher = {
            'dbscan': DBSCAN,
            'kmeans': KMeans,
            'optics': OPTICS
        }
        if HDBSCAN_INSTALLED:
            clustering_dispatcher['hdbscan'] = HDBSCAN

        if clustering in clustering_dispatcher:
            self.clustering_class = clustering_dispatcher[clustering](
                **params.get('clustering', {})
            )
        elif isinstance(clustering, ClusterMixin):
            self.clustering_class = clustering(**params.get('clustering', {}))
        else:
            raise ValueError('clustering has to be one of the available '
                             'clusters or a sklearn ClusterMixin.')

        if clustering_random_state and hasattr(self.clustering_class,
                                               'random_state'):
            self.clustering_class.random_state = clustering_random_state

        self.embedded_points = None
        self.reduced_points = None
        self.cluster_ids = None
        self.cluster_names = None
        self.cluster_kws = None
        self.kw_dictionary = {}
        self.silhouette = None
        self.optimise_results = {}

        self.embedded_points_filename = 'embedded_points.npy'
        self.reduced_points_filename = 'reduced_points.npy'
        self.vectorizer_filename = 'vectorizer.pkl'
        self.reducer_filename = 'reducer.pkl'
        self.clustering_filename = 'clustering.pkl'

    def fit(self, X, *_):
        """
        Fits all clusters in the pipeline

        Args:
            X: A list of texts

        Returns:
            A TextClustering object

        """
        self.fit_step(X, step='vectorizer')
        self.fit_step(step='reducer')
        self.fit_step(step='clustering')

        if self.embedding == 'tf-idf' and self.n_kw:
            self._find_keywords(self.embedded_points.toarray(), n_kw=self.n_kw)

        return self

    def fit_step(self, X=None, y=None, step='vectorizer'):
        """Internal function for partial fitting only a certain step"""
        if step == 'vectorizer':
            self.embedded_points = self.vectorizer.fit_transform(X)
        elif step == 'reducer':
            if self.embedded_points is None:
                raise ValueError(
                    'You must embed/vectorise the points before reducing dimensionality'
                )
            if X is None:
                X = self.embedded_points

            self.reduced_points = self.reducer_class.fit_transform(X=X, y=y)
        elif step == 'clustering':
            points = (
                self.reduced_points if self.cluster_reduced else
                self.embedded_points
            )
            self.clustering_class.fit(points)
            self.cluster_ids = self.clustering_class.labels_

    def optimise(self, X, param_grid, n_cluster_range=None, max_noise=0.2,
                 verbose=False):
        """
        Optimises the clustering silhouette based on a parameter grid,
        a range on number of clusters and a range on noise.

        It is customised to avoid re-fitting of intermediate
        steps (vectorizer and reducer) more than necessary.

        Args:
            X (iterable[str]): A list of texts to be clustered

            param_grid (dict) : A parameter grid, example:
                param_grid = {'reducer': {'min_dist': [0.0, 0.2], 'n_neighbors': [2,3,5],
                'metric': ['cosine', 'euclidean']},
                'clustering': {'min_samples': [2, 5], 'eps': [0.5, 1, 1.5]}}

            n_cluster_range (2-uple of ints): A 2-uple describing the max and min number
                of clusters (e.g.: (10, 20)). If unset, will just choose the best silhouette

            max_noise (float in [0,1]): The maximum fraction of points unclustered. Default: 0.2

        Returns:
            dict: A dictionary of results.

            The function returns a dictionary containing  "params_list", "silhouette"
            (the silhouette for each parameter) and "best_clustering"
            (the best clustering parameters)

        """

        min_n_clusters = (n_cluster_range[0] if n_cluster_range else 0)
        max_n_clusters = (n_cluster_range[1] if n_cluster_range else 10**5)

        # X might be transformed to be a vector, so we need to save the input
        # texts

        X_text = X

        # Linearises Dictionary to be compatible with grid search so it
        # becomes one dictionary with 'step__parameter'
        if self.reducer == 'tsne':
            logger.warning("TSNE is not suitable for predicting on new data."
                           "Skipping Vectoriser/TSNE optimisation parameters")
            self.fit(X)
            X = self.reduced_points

            pipeline = Pipeline([('clustering', self.clustering_class)],
                                memory=CACHE_DIR)

            params = {}

        elif self.cluster_reduced:
            # You cannot pickle sparse uMAP with more than 4096 points.
            # See https://github.com/lmcinnes/umap/issues/674
            # Until that issue is fixed, we need to convert everything to dense or cannot cache
            # the transformations of the pipeline

            memory = (CACHE_DIR if len(X) < 4096 or self.embedding != 'tf-idf' else None)
            pipeline = Pipeline([('vectorizer', self.vectorizer),
                                 ('reducer', self.reducer_class),
                                 ('clustering', self.clustering_class)],
                                memory=memory)

            params = {
                **{f'reducer__{key}': value for key, value in
                   param_grid.get('reducer', {}).items()}
            }

            params = {
                **params,
                **{f'vectorizer__{key}': value for key, value in
                   param_grid.get('vectorizer', {}).items()}
            }

        else:
            self.vectorizer.cache_transformed = True
            pipeline = Pipeline([('vectorizer', self.vectorizer),
                                 ('clustering', self.clustering_class)],
                                memory=CACHE_DIR)
            params = {
                **{f'vectorizer__{key}': value for key, value in
                   param_grid.get('vectorizer', {}).items()}
            }

        params = {
            **params,
            **{f'clustering__{key}': value for key, value in
               param_grid.get('clustering', {}).items()}
        }

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            scoring={
                'silhouette': _clustering_score,
                'noise': _clustering_noise,
                'n_clusters': _number_of_clusters
            },
            refit='silhouette'
        )

        logging_level = logger.level
        if verbose <= 1:
            # Previously disable logging to allow the loading bar to run
            # uninterruptly. Will reset after.
            logging.getLogger().setLevel(logging.WARNING)
            logger.setLevel(logging.WARNING)

        # Prunes result to actually optimise under constraints
        best_silhouette = 0
        best_params = {}

        grid.fit(X, y=None)

        for params, silhouette, noise, n_clusters in zip(
                grid.cv_results_['params'],
                grid.cv_results_['mean_test_silhouette'],
                grid.cv_results_['mean_test_noise'],
                grid.cv_results_['mean_test_n_clusters']
        ):

            if min_n_clusters <= n_clusters <= max_n_clusters\
                    and noise <= max_noise\
                    and silhouette > best_silhouette:
                best_silhouette = silhouette
                best_params = params

        if not best_params:
            logger.warning("Could not find any clustering model with the "
                           "specified number of clusters and noise")

        self.silhouette = best_silhouette
        self.optimise_results = {
            key: value for key, value in grid.cv_results_.items()
            if key[:5] != 'split'  # We don't need all cross-val split results
        }

        self.set_params(best_params, from_parameter_grid=True)
        # Fits the pipeline again with the best parameters
        self.fit(X_text)

        logger.setLevel(logging_level)

        return best_params

    def save(self, folder, components='all', create_folder=True):
        """
        Saves the different steps of the pipeline

        Args:
            folder(str): path to folder
            components(list or 'all'): List of components to save. Options are: 'embbedded_points',
            'reduced_points', 'vectorizer', 'reducer', and 'clustering_model'. By default, loads
            'all' (you can get all components by listing the class param
             TextClustering.components)

        """
        if create_folder:
            os.makedirs(folder, exist_ok=True)

        if components == 'all' or 'embedded_points' in components:
            np.save(os.path.join(folder, self.embedded_points_filename), self.embedded_points)

        if components == 'all' or 'reduced_points' in components:
            np.save(os.path.join(folder, self.reduced_points_filename), self.reduced_points)

        if components == 'all' or 'vectorizer' in components:
            with open(os.path.join(folder, self.vectorizer_filename), 'wb') as f:
                pickle.dump(self.vectorizer, f)

        if components == 'all' or 'reducer' in components:
            with open(os.path.join(folder, self.reducer_filename), 'wb') as f:
                pickle.dump(self.reducer_class, f)

        if components == 'all' or 'clustering_model' in components:
            with open(os.path.join(folder, self.clustering_filename), 'wb') as f:
                pickle.dump(self.clustering_class, f)

    def load(self, folder, components='all'):
        """
        Loads the different steps of the pipeline

        Args:
            folder(str): path to folder
            components(list or 'all'): List of components to load. Options are: 'embbedded_points',
            'reduced_points', 'vectorizer', 'reducer', and 'clustering_model'. By default, loads
            'all' (you can get all components by listing the class param
             TextClustering.components)

        """

        if components == 'all' or 'embedded_points' in components:
            self.embedded_points = np.load(os.path.join(folder, self.embedded_points_filename),
                                           allow_pickle=True)
            if not self.embedded_points.shape:
                self.embedded_points = self.embedded_points[()]

        if components == 'all' or 'reduced_points' in components:
            self.reduced_points = np.load(os.path.join(folder, self.reduced_points_filename),
                                          allow_pickle=True)

        if components == 'all' or 'vectorizer' in components:
            with open(os.path.join(folder, self.vectorizer_filename), 'rb') as f:
                self.vectorizer = pickle.load(f)

        if components == 'all' or 'reducer' in components:
            with open(os.path.join(folder, self.reducer_filename), 'rb') as f:
                self.reducer_class = pickle.load(f)

        if components == 'all' or 'clustering_model' in components:
            with open(os.path.join(folder, self.clustering_filename), 'rb') as f:
                self.clustering_class = pickle.load(f)

    def stability(self):
        """Function to calculate how stable the clusters are"""
        raise NotImplementedError

    def break_cluster(self, cluster_id):
        """Breaks a specific by re-clustering a subset"""
        raise NotImplementedError

    def merge_clusters(self, dictionary):
        """Merge cluster based on ids"""
        self.cluster_ids = [dictionary.get(cluster_id, cluster_id) for
                            cluster_id in self.cluster_ids]

    def set_params(self, params, from_parameter_grid=False):
        """
        Set parameters of clustering, reducer and vectorizer

        Args:
            params: A dictionary of parameters
            from_parameter_grid: Whether the dictionary is in the form of
            sklearn parameter grid (e.g. {'clustering__eps': 1}), or a two
            level dictionary (e.g. {'clustering': {'eps': 1}}) (default).

        Returns:
            None

        """
        if from_parameter_grid:
            # This is to convert from a sklearn parameter to a two-level
            # dictionary, which is what we work with
            new_params = defaultdict(dict)

            for key, value in params.items():
                level_1_key, level_2_key = key.split('__')
                new_params[level_1_key][level_2_key] = value

            params = new_params

        self.vectorizer.set_params(**params.get('vectorizer', {}))
        self.reducer_class.set_params(**params.get('reducer', {}))
        self.clustering_class.set_params(**params.get('clustering', {}))

    def _find_keywords(self, X_transformed, n_kw=10):
        if self.embedding != 'tf-idf':
            raise ValueError(f'Cannot find keyword for '
                             f'{self.embedding} vectorizer')
        idx_to_word = {value: key for key, value in
                       self.vectorizer.vectorizer.vocabulary_.items()}

        # Initialise a dictionary to calculate the aggregated vector for each
        # cluster

        aggregated = {key: np.zeros(X_transformed.shape[1])
                      for key in self.cluster_ids}

        for i in range(len(X_transformed)):
            aggregated[self.cluster_ids[i]] += X_transformed[i]

        self.kw_dictionary = {
            key: ','.join(_find_words_from_frequency(vector,
                                                     idx_to_word,
                                                     n_kw=n_kw))
            for key, vector in aggregated.items()
        }

        self.cluster_kws = [
            self.kw_dictionary.get(cluster_id) for cluster_id in
            self.cluster_ids
        ]


def _find_words_from_frequency(vector, idx_to_word, n_kw=5):
    """
    Given a vector of frequencies and a dictionary connecting indexes
     to word, find most prominent words
    up to n_kw, eliminating possible duplicates or sub-words.

    Args:
        vector: A vector representing frequencies (or any other scores) for
        each word.
        idx_to_word: A dictionary mapping indices in `vector` to words.
        n_kw: number of "prominent" words to output.

    Returns:
        A list of n_kw words (strings).

    """
    sorted_idx = np.argsort(vector)[::-1]
    # Tests if new words do not overlap with existing ones (this is to
    # exclude things like: 'cluster,clustering,clustered')

    current_kw_list = []
    i = 0

    while len(current_kw_list) < n_kw and i <= len(vector)-1:
        new_word = idx_to_word[sorted_idx[i]]
        used_word = False
        for kw in current_kw_list:
            if new_word in kw or kw in new_word:
                used_word = True
                break

        if not used_word:
            current_kw_list += [new_word]

        i += 1

    return current_kw_list


def _number_of_clusters(estimator, X, y=None):
    """Utility to return number of clusters, excluding noise (if exists)"""
    label_set = set(estimator['clustering'].labels_)
    return len(label_set)-1*(-1 in label_set)


def _clustering_noise(estimator, X, y=None):
    """Returns the clustering noise (predictions = -1)"""
    return len([x for x in estimator['clustering'].labels_ if x == -1]) /\
        len(estimator['clustering'].labels_)


def _clustering_score(estimator, X, y=None):
    """
    Scores a clustering based on the silhouette score

    Args:
        estimator: A sklearn Pipeline. Assumes the estimator has a set
        called 'embedding' and a step called 'clustering', which is a
        sklearn.base.ClusterMixin

        X: A vector of size n x m, where n is the number of points and m
         the dimension of each point

    Returns:
        The average silhouette score of X, given the estimator labels,
         or 0 if only one cluster has been found

    """
    # If has a reducer, uses the embeddings
    if hasattr(estimator.named_steps, 'reducer'):
        X_points = estimator.named_steps['reducer'].embedding_
    # If it does not, uses the vectorizer
    elif estimator.named_steps.get('vectorizer'):
        if hasattr(estimator.named_steps['vectorizer'], 'X_transformed'):
            X_points = estimator.named_steps['vectorizer'].X_transformed
        else:
            X_points = estimator['vectorizer'].transform(X)
    # If no reducer of vectorizers are present, it must be that X has
    # already been reduced
    else:
        X_points = X

    predicted_labels = estimator['clustering'].labels_

    return silhouette_score(X_points, predicted_labels) \
        if 2 <= len(np.unique(predicted_labels)) <= len(predicted_labels)-1 \
        else 0
