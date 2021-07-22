import pandas as pd
from wellcomeml.ml.clustering import TextClustering
from wellcomeml.viz.visualize_cluster import visualize_clusters

url = "https://datalabs-public.s3.eu-west-2.amazonaws.com/" \
      "datasets/epmc/random_sample.csv"
data = pd.read_csv(url)

text = list(data['text'])

clustering = TextClustering(embedding='tf-idf', reducer='umap', params={
    'reducer': {'min_dist': 0.1, 'n_neighbors': 10},
    'vectorizer': {'min_df': 0.0002},
    'clustering': {'min_samples': 20, 'eps': 0.2}
})

clustering.fit(text)

visualize_clusters(clustering, 0.05, 0.8, output_in_notebook=False,
                   output_file_path="test.html")
