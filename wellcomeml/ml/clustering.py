import umap

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

from wellcomeml.ml import Vectorizer


class ClusterPipeline(object):
    def __init__(self, embedding='tf-idf', reducer='umap', clustering='dbscan',
                 params={}):
        """
        Initialises a cluster pipeline

        Args:
            embedding(str): 'tf-idf', 'bert', 'keras' or 'doc2vec'.
            reducer(str): 'umap' or 'tsne'
            clustering(str): Only 'dbscan' allowed for now
            params(dict): Dictionary containing extra parameters for
            embedding, reducer or clustering models. Example:
            {'clustering': {'eps': 0.1}, 'embedding': {'ngram_range': (1,2)}}
        """
        self.embedding = Vectorizer(embedding=embedding,
                                    **params.get('embedding', {}))

        reducer_dispatcher = {
            'umap': umap.UMAP,
            'tsne': TSNE
        }
        self.reducer = reducer_dispatcher[reducer](**params.get('reducer'))

        clustering_dispatcher = {
            'dbscan': DBSCAN
        }

        self.clustering = clustering_dispatcher[clustering](
            **params.get('clustering')
        )

    def find_best_silhouette(self, param_grid):
        pass

    
