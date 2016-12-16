"""Helper functions
"""
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import metrics

import gzip
import json
import time
import pickle
import os
import logging

import warnings
warnings.filterwarnings("ignore")


logs_that_exist = {}

def getLogger(name='cs505.log'):
    log = logging.getLogger(name)

    if name in logs_that_exist:
        return log

    log.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(name)
    file_handler.setFormatter(log_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)

    log.addHandler(file_handler)
    log.addHandler(stream_handler)

    logs_that_exist[name] = True

    return log

LOG = getLogger()

# Load the review data
def json_from_gzip_file(file_name):
    reviews = []
    with gzip.open(file_name) as gz_file:
        for jsonstr in gz_file:
            review = json.loads(jsonstr.decode('utf-8'))
            reviews.append(review)

    return reviews

def list_to_pd(mylist):
    return pd.DataFrame(mylist)

def gzip_json_to_pd(file_name):
    return list_to_pd(json_from_gzip_file(file_name))

def print_df(df):
    LOG.info(df.to_string())

def export_pickle(data, file_name, root='data'):
    with open(os.path.join(root, file_name), 'wb') as f:
        pickle.dump(data, f)

def import_pickle(file_name, root='data'):
    with open(os.path.join(root, file_name), 'rb') as f:
        return pickle.load(f)

def describe_cluster(cluster):
    LOG.info("======================================================")
    LOG.info("Unclustered Features")
    LOG.info("Cluster label: %i" % cluster['cluster_labels'].mean())
    LOG.info("Size of cluster: %i" % cluster.shape[0])
    LOG.info("Avg review rating: %f" % cluster['overall'].mean())
    LOG.info("Random sample:\n%s" % cluster.sample().to_string())

    LOG.info("Clustered Features")
    LOG.info("Avg review length: %i" % cluster['review_length'].mean())
    LOG.info("Avg polarity: %f" % cluster['polarity'].mean())
    LOG.info("Avg subjectivity: %f" % cluster['subjectivity'].mean())
    LOG.info("Avg helpfulness: %f" % cluster['percent_helpful'].mean())
    LOG.info("Avg reviewer's rating: %f" % cluster['reviewers_average_rating'].mean())
    LOG.info("Avg reviewer's max reviews on any day: %f" % cluster['reviewers_max_reviews_on_any_day'].mean())
    LOG.info("Avg reviewer's total num reviews: %f" % cluster['reviewers_total_num_reviews'].mean())
    LOG.info("Avg reviewer's reviews per day: %f" % cluster['reviewers_reviews_per_day'].mean())

def bench_kmeans(estimator, name, data):
    LOG.info('% 9s' % 'init n_clusters time  inertia (lower is better) silhouette (lower is better)')
    t0 = time.time()
    estimator.fit(data)
    LOG.info('% 9s   %i   %.2fs    %i' % (name, len(estimator.cluster_centers_), (time.time() - t0), estimator.inertia_))
