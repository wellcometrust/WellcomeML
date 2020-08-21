from copy import deepcopy

import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import umap

from wellcomeml.ml import Vectorizer


class TextClustering(object):
    """
    Class

    Attributes:
        vectorizer: The embedding Vectorizer object
        reducer: A dimensionality reduction object
        clustering: A clustering model object
        cluster_ids: Ids of the cluster
        cluster_names: Names of the clusters
        cluster_kws: Keywords for the clusters (only if embedding=tf-idf)
    """
    def __init__(self, embedding='tf-idf', reducer='umap', clustering='dbscan',
                 n_kw=10, params={}):
        """
        Initialises a cluster pipeline

        Args:
            embedding(str): 'tf-idf', 'bert', 'keras' or 'doc2vec'.
            reducer(str): 'umap' or 'tsne'
            clustering(str): Only 'dbscan' allowed for now
            n_kw(int): If n_kw > 0 and embedding='tf-idf', the model
            outputs, besides a cluster id, a list of keyword for each
            cluster.
            params(dict): Dictionary containing extra parameters for
            embedding, reducer or clustering models. Example:
            {'clustering': {'eps': 0.1}, 'embedding': {'ngram_range': (1,2)}}
        """
        self.embedding = embedding
        self.vectorizer = Vectorizer(embedding=embedding,
                                     **params.get('embedding', {}))
        self.n_kw = n_kw

        reducer_dispatcher = {
            'umap': umap.UMAP,
            'tsne': TSNE
        }
        self.reducer = reducer_dispatcher[reducer](**params.get('reducer', {}))

        clustering_dispatcher = {
            'dbscan': DBSCAN
        }

        self.clustering = clustering_dispatcher[clustering](
            **params.get('clustering', {})
        )

        self.cluster_ids = None
        self.cluster_names = None
        self.cluster_kws = None

    def find_best_silhouette(self, param_grid):
        pass

    def fit(self, X, *_):
        """
        Fits all clusters in the pipeline

        Args:
            X: A list of texts

        Returns:
            A TextClustering object

        """
        self.embedded_points = self.vectorizer.fit_transform(X)
        self.reduced_points = self.reducer.fit_transform(self.embedded_points)
        self.clustering.fit(self.reduced_points)
        self.cluster_ids = self.clustering.labels_

        if self.embedding == 'tf-idf' and self.n_kw:
            self._find_keywords(self.embedded_points.toarray(), n_kw=self.n_kw)

        return self

    def optimise(self, X, param_grid, n_cluster_range, noise_range):
        """Optimises the clustering given a parameter grid """

        min_n_clusters = 0
        max_n_clusters = 10**5

        parameter_grid_object = ParameterGrid(param_grid=param_grid)

        best_silhouette = -1
        best_clustering = None

        params_list = []
        best_params = None

        for params in parameter_grid_object:
            self.set_params(params)
            self.fit(X)

            silhouette = _clustering_score(self.clustering,
                                           self.reduced_points)
            n_clusters = len(np.unique(self.clustering.labels_))
            noise = sum(self.clustering.labels_ == -1)

            # If there is noise, the label "-1" is added as a
            # valid cluster number.
            if noise > 0:
                n_clusters -= 1

            params_list += [{
                "params": params,
                "silhouette": silhouette,
                "n_clusters": n_clusters,
                "noise": noise
            }]

            if silhouette > best_silhouette and min_n_clusters <= \
                    n_clusters <= max_n_clusters:
                best_silhouette = silhouette
                best_params = deepcopy(params)

        self.set_params(best_params)

        return {
            "params_list": params_list,
            "silhouette": best_silhouette,
            "clustering": best_clustering
        }

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

    def set_params(self, params):
        self.vectorizer.set_params(**params['vectorizer'])
        self.reducer.set_params(**params['reducer'])
        self.clustering.set_params(**params['clustering'])

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

        kw_dictionary = {
            key: ','.join(_find_words_from_frequency(aggregated,
                                                     idx_to_word,
                                                     n_kw=n_kw))
            for key, vector in aggregated.items()
        }

        self.cluster_kws = [
            kw_dictionary.get(cluster_id) for cluster_id in self.cluster_ids
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
    import pdb
    pdb.set_trace()
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


def _clustering_score(estimator, X, y=None):
    """
    Scores a clustering based on the silhouette score
    (function suitable to use with GridSearchCV)

    Args:
        estimator: A sklearn estimator. Assumes the estimator has
         labels_ with size equals to len(X)
        X: A vector of size n x m, where n is the number of points and m
         the dimension of each point

    Returns:
        The average silhouette score of X, given the estimator labels,
         or 0 if only one cluster has been found

    """
    predicted_labels = estimator.labels_

    return silhouette_score(X, predicted_labels) if \
        len(np.unique(predicted_labels)) > 1 else 0
