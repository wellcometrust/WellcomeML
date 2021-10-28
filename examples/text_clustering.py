from wellcomeml.ml.clustering import TextClustering

# This is bad clustering

cluster = TextClustering(
    reducer='umap', n_kw=2,
    params={'clustering': {'min_samples': 2, 'eps': 3}}
)

X = ['Wellcome Trust',
     'The Wellcome Trust',
     'Sir Henry Wellcome',
     'Francis Crick',
     'Crick Institute',
     'Francis Harry Crick']

cluster.fit(X)
print("Not very good clusters:")
print([(x, cluster) for x, cluster in zip(X, cluster.cluster_ids)])

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
