#!/usr/bin/env python
import numpy as np
import scipy as sp
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA

from textblob import TextBlob

import gzip
import json
import time
import datetime
import pickle
import os
from argparse import ArgumentParser
from cs505_helper_funcs import json_from_gzip_file, list_to_pd, gzip_json_to_pd
from cs505_helper_funcs import print_df, describe_cluster, getLogger
from cs505_helper_funcs import export_pickle, import_pickle
from cs505_helper_funcs import bench_kmeans

from cs505_gen_graphs import plot_kmeans, plot_kmeans2, plot_total_reviews, plot_helpfulness_of_reviews

import warnings
warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("--local", action='store_true')
args = parser.parse_args()

# Configure the analysis
if args.local:
    BEAUTY_REVIEWS_FILE = "res/reviews_Beauty_5.json.gz"
else:
    BEAUTY_REVIEWS_FILE = "res/reviews_Beauty.json.gz" # Beauty (2,023,070 reviews)

if args.local:
    ELEC_REVIEWS_FILE = "res/reviews_Electronics-small.json.gz" # Electronics Subset (781,889 reviews)
else:
    ELEC_REVIEWS_FILE = "res/reviews_Electronics.json.gz"

BABY_REVIEWS_FILE = "res/reviews_Baby.json.gz" # Baby (915,446 reviews)
SMALL_FILE = "res/small.json.gz"

if args.local:
    REVIEWS_FILE = BEAUTY_REVIEWS_FILE
else:
    REVIEWS_FILE = ELEC_REVIEWS_FILE

START_YEAR = 1997
END_YEAR = 2013

LOG = getLogger()

# Parts of a review:
#     reviewerID     - ID of the reviewer, e.g. A2SUAM1J3GNN3B
#     asin           - ID of the product, e.g. 0000013714
#     reviewerName   - name of the reviewer
#     helpful        - helpfulness rating of the review, e.g. 2/3
#     reviewText     - text of the review
#     overall        - rating of the product
#     summary        - summary of the review
#     unixReviewTime - time of the review (unix time)
#     reviewTime     - time of the review (raw)
LOG.info("Loading %s...", REVIEWS_FILE)
if 'reviews.pkl' in os.listdir('data'):
    reviews = import_pickle('reviews.pkl')
else:
    reviews = gzip_json_to_pd(REVIEWS_FILE)
    export_pickle(reviews, 'reviews.pkl')


LOG.info("Splitting data by year...")
if 'reviews-split-by-year.pkl' in os.listdir('data'):
    reviews = import_pickle('reviews-split-by-year.pkl')
else:
    reviews['year_written'] = reviews.unixReviewTime.apply(lambda x: time.gmtime(x)[0])
    # Throw out incomplete years for better analysis
    reviews = reviews[(reviews.year_written >= START_YEAR) & (reviews.year_written <= END_YEAR)]
    export_pickle(reviews, 'reviews-split-by-year.pkl')


LOG.info("Adding features...")
# Review Centric Features
if 'reviews-with-review-centric-features.pkl' in os.listdir('data'):
    reviews = import_pickle('reviews-with-review-centric-features.pkl')
else:
    reviews['helpful_votes'] = reviews.helpful.apply(lambda x: x[0])
    reviews['total_votes'] = reviews.helpful.apply(lambda x: x[1])
    reviews['percent_helpful'] = reviews['helpful_votes'] / reviews['total_votes']
    reviews['percent_helpful'] = reviews['percent_helpful'].fillna(.5)
    LOG.info("Successfully added `percent_helpful` feature")

    # Perform polarity analysis of the review text, ranges from -1 to 1
    reviews['polarity'] = reviews['reviewText'].apply(lambda x: TextBlob(x).sentiment.polarity)
    # Perform subjective analysis of the review text, ranges from 0 to 1
    reviews['subjectivity'] = reviews['reviewText'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    LOG.info("Successfully added `polarity` and `subjectivity` features")

    reviews['review_length'] = reviews['reviewText'].str.split().apply(len)
    LOG.info("Successfully added `review_length` feature")
    export_pickle(reviews, 'reviews-with-review-centric-features.pkl')

# Reviewer Centric Features
if 'reviews-with-reviewer-centric-features.pkl' in os.listdir('data'):
    reviews = import_pickle('reviews-with-reviewer-centric-features.pkl')
else:
    reviewers_avg_rating = reviews.groupby("reviewerID", as_index=False)['overall'].mean()
    reviews = pd.merge(reviews, reviewers_avg_rating, on="reviewerID")
    reviews.rename(columns = {'overall_y':'reviewers_average_rating'}, inplace=True)
    LOG.info("Successfully added `reviewers_average_rating` feature")

    group_by_id_time = reviews.groupby(["reviewerID", "reviewTime"])["reviewerID"].agg({'overall_x': 'count'})
    mask = group_by_id_time.groupby(level=0).agg('idxmax')
    df_count = group_by_id_time.loc[mask['overall_x'].tolist()]
    df_count = df_count.reset_index()
    df_count.rename(columns={"overall_x" : "reviewers_max_reviews_on_any_day"}, inplace=True)
    df_count.drop("reviewTime", axis=1, inplace=True)
    reviews = pd.merge(reviews, df_count, on="reviewerID")
    reviews.rename(columns={"overall_x": "overall"}, inplace=True)
    LOG.info("Successfully added `reviewers_max_reviews_on_any_day` feature")

    group_by_id = reviews.groupby("reviewerID").count()
    group_by_id["reviewerID"] = group_by_id.index
    group_by_id = group_by_id[["overall", "reviewerID"]]
    group_by_id.rename(columns={"overall_y" : "reviewers_total_num_reviews"}, inplace=True)
    reviews = pd.merge(reviews, group_by_id, on="reviewerID")
    reviews.rename(columns={"overall_x":"overall", "overall_y": "reviewers_total_num_reviews"}, inplace=True)
    LOG.info("Successfully added `reviewers_total_num_reviews` feature")

    reviews["reviewers_reviews_per_day"] = reviews["reviewers_total_num_reviews"] / 365.0
    LOG.info("Successfully added `reviewers_reviews_per_day` feature")
    # TODO(eugenek): `reviewers_reviews_similarity`
    export_pickle(reviews, 'reviews-with-reviewer-centric-features.pkl')

LOG.info("Printing 10 entries off the head of the reviews...")
print_df(reviews.head(10))

LOG.info("Creating features DF...")
if 'features.pkl' in os.listdir('data'):
    features = import_pickle('features.pkl')
else:
    features = pd.DataFrame()
    features = features.append(reviews['reviewers_reviews_per_day'])
    features = features.append(reviews['reviewers_total_num_reviews'])
    features = features.append(reviews['reviewers_max_reviews_on_any_day'])
    features = features.append(reviews['reviewers_average_rating'])
    features = features.append(reviews['polarity'])
    features = features.append(reviews['subjectivity'])
    features = features.append(reviews['percent_helpful'])
    features = features.append(reviews['review_length'])
    features = features.transpose()
    export_pickle(features, 'features.pkl')

# TODO(eugenek): Dead for now...
# MIN_HELPFUL_VOTES = 3
# PERCENT_TO_BE_HELPFUL = .7
# reviews['is_helpful'] = np.where((reviews.percent_helpful > PERCENT_TO_BE_HELPFUL) &
#                                  (reviews.helpful_votes > MIN_HELPFUL_VOTES), "Yes", "No")


LOG.info('Peforming kmeans clustering...')
n_samples, n_features = features.shape
LOG.info('n_samples: %s, n_features: %s', str(n_samples), str(n_features))
LOG.info('Evaluating best kmeans parameters...')

# for i in range(1, 7):
#     n_clusters = 2**i
#     bench_kmeans(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10), name="k-means++", data=features)

# Derived from above
best_n_clusters = 8

LOG.info('Scaling features...')
# Scale everything to be on the scale of 0 to 1
features['reviewers_reviews_per_day'] = features['reviewers_reviews_per_day'] / features['reviewers_reviews_per_day'].max()
features['reviewers_total_num_reviews'] = features['reviewers_reviews_per_day'] / features['reviewers_total_num_reviews'].max()
features['reviewers_max_reviews_on_any_day'] = features['reviewers_max_reviews_on_any_day'] / features['reviewers_max_reviews_on_any_day'].max()
features['reviewers_average_rating'] = features['reviewers_average_rating'] / 5.0
features['polarity'] = (features['polarity'] - -1) / (1 - -1)
features['subjectivity'] = features['subjectivity']
features['percent_helpful'] = features['percent_helpful']
features['review_length'] = features['review_length'] / features['review_length'].max()

LOG.info('Clustering with kmeans...')

if 'kmeans.pkl' in os.listdir('data'):
    kmeans_per_year = import_pickle('kmeans.pkl')
else:
    features_with_year = features.transpose()
    features_with_year = features_with_year.append(reviews['year_written'])
    features_with_year = features_with_year.transpose()
    kmeans_per_year = []
    for year in range(START_YEAR+7, END_YEAR+1):
        kmeans = KMeans(init='k-means++', n_clusters=best_n_clusters)
        kmeans_features = features[(features_with_year.year_written == year)]
        kmeans.fit_predict(kmeans_features)
        kmeans_per_year.append(kmeans)
    export_pickle(kmeans_per_year, 'kmeans.pkl')

LOG.info("Describing clusters...")
for year, kmeans in zip(range(START_YEAR+7, END_YEAR+1), kmeans_per_year):
    reviews.loc[reviews['year_written'] == year, 'cluster_labels'] = kmeans.labels_

for year in range(START_YEAR+7, END_YEAR+1):
     cluster_by_year = reviews[(reviews.year_written == year)]
     for cluster_label in range(0, best_n_clusters):
         describe_cluster(cluster_by_year[cluster_by_year['cluster_labels'] == cluster_label])

LOG.info("Plotting results...")
LOG.info("Computing fake latitude, and fake longitude to plot with...")
scaler = PCA(n_components=2).fit(features)
fake_coords = scaler.transform(features).transpose()
fake_lat, fake_lng = fake_coords[0], fake_coords[1]
reviews['fake_lat'] = fake_lat
reviews['fake_lng'] = fake_lng
for year, kmeans in zip(range(START_YEAR+7, END_YEAR+1), kmeans_per_year):
    plot_kmeans(kmeans, scaler, reviews[reviews['year_written'] == year], fig_name='kmeans-clustering-'+str(year))

plot_kmeans2(kmeans_per_year, scaler, reviews, fig_name='kmeans-clustering-four-years.png')

plot_total_reviews(reviews)

plot_helpfulness_of_reviews(reviews)

LOG.info("Done!")
