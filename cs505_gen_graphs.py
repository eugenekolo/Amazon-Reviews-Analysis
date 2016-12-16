#!/usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

import os

from cs505_helper_funcs import getLogger

import warnings
warnings.filterwarnings("ignore")

LOG = getLogger()

def plot_kmeans(kmeans, scaler, data, fig_name='kmeans-clustering.png'):
    LOG.info("Plotting clusters...")

    colors = [
        '#4EACC5', '#FF9C34', '#123123', '#FEED12', '#321ee1', '#20FdFd', '#444444', '#076E55',
        '#4EFFC5', '#F22C34', '#1DD123', '#FEBB12', '#421323', '#F220F0', '#33ff44', '#9d0f6b',
        '#4EACFF', '#F29CC4', '#1EE123', '#FEDD12', '#32ff21', '#F6F0F0', '#114444', '#33ee33',
        '#4BBCC5', '#737d02', '#FFF12F', '#FE1112', '#322221', '#F750F0', '#224444', '#ee3333',
    ]

    labels = []
    fig, ax = plt.subplots(1)
    for cluster_label, col in zip(range(len(kmeans.cluster_centers_)), colors):
        cluster = data[data['cluster_labels'] == cluster_label]
        # Plot reviews
        sns.regplot(cluster['fake_lat'], cluster['fake_lng'],
            color=col, marker='.', scatter=True, fit_reg=False)

        # Plot centroids
        cluster_center = kmeans.cluster_centers_[cluster_label]
        sns.regplot(scaler.transform(cluster_center).transpose()[0], scaler.transform(cluster_center).transpose()[1],
            color=col, scatter_kws={'s':100, 'linewidth':'1' ,'edgecolors':'k'}, scatter=True, fit_reg=False)

        # Add a label
        helpfulness = cluster['percent_helpful'].mean()
        reviewers_avg_rating =  cluster['reviewers_average_rating'].mean()
        reviewers_reviews_per_day =cluster['reviewers_reviews_per_day'].mean()
        label = "cluster%s\nhelpfulness=%.2f\nreviewer's rating=%.2f\ndaily reviews=%.2f" % \
            (cluster_label, helpfulness, reviewers_avg_rating, reviewers_reviews_per_day)
        labels.append(mpatches.Patch(color=col, label=label))


    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    sns.set(font_scale=.75)
    ax.legend(bbox_to_anchor=(0,-.125), loc=3, ncol=4, handles=labels)
    fig.savefig(fig_name)
    sns.set(font_scale=1)

def plot_kmeans2(kmeans_per_year, scaler, data, fig_name='kmeans-clustering.png'):
    LOG.info("Plotting clusters...")

    colors = [
        '#4EACC5', '#FF9C34', '#123123', '#FEED12', '#321ee1', '#20FdFd', '#444444', '#076E55',
        '#4EFFC5', '#F22C34', '#1DD123', '#FEBB12', '#421323', '#F220F0', '#33ff44', '#9d0f6b',
        '#4EACFF', '#F29CC4', '#1EE123', '#FEDD12', '#32ff21', '#F6F0F0', '#114444', '#33ee33',
        '#4BBCC5', '#737d02', '#FFF12F', '#FE1112', '#322221', '#F750F0', '#224444', '#ee3333',
    ]


    fig, axes = plt.subplots(4,1)
    plt.subplots_adjust(hspace=.5)
    fig.set_size_inches(8, 11)
    kmeans_list = []
    kmeans_list.append(kmeans_per_year[0]) # 2004
    kmeans_list.append(kmeans_per_year[4]) # 2008
    kmeans_list.append(kmeans_per_year[8]) # 2012
    kmeans_list.append(kmeans_per_year[9]) # 2013
    for ax, kmeans, year in zip(axes.flatten(), kmeans_list, [2004, 2008, 2012, 2013]):
        years_data = data[data['year_written'] == year]
        labels = []
        for cluster_label, col in zip(range(len(kmeans.cluster_centers_)), colors):
            cluster = years_data[years_data['cluster_labels'] == cluster_label]
            # Plot reviews
            sns.regplot(cluster['fake_lat'], cluster['fake_lng'],
                color=col, marker='.', scatter=True, fit_reg=False, ax=ax)

            # Plot centroids
            cluster_center = kmeans.cluster_centers_[cluster_label]
            sns.regplot(scaler.transform(cluster_center).transpose()[0], scaler.transform(cluster_center).transpose()[1],
                color=col, scatter_kws={'s':100, 'linewidth':'1' ,'edgecolors':'k'}, scatter=True, fit_reg=False, ax=ax)

            # Add a label
            helpfulness = cluster['percent_helpful'].mean()
            reviewers_avg_rating =  cluster['reviewers_average_rating'].mean()
            reviewers_reviews_per_day = cluster['reviewers_reviews_per_day'].mean()
            subjectivity = cluster['subjectivity'].mean()
            label = "cluster%s\nhelpfulness=%.2f\nreviewer's rating=%.2f\ndaily reviews=%.2f\nsubjectivity=%.2f" % \
                (cluster_label, helpfulness, reviewers_avg_rating, reviewers_reviews_per_day, subjectivity)
            labels.append(mpatches.Patch(color=col, label=label))

            sns.set(font_scale=.75)
            ax.set_title("%s Review Clustering" % year)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.legend(bbox_to_anchor=(0,-.355), loc=3, ncol=4, handles=labels)

    fig.savefig(fig_name)
    sns.set(font_scale=1)

def plot_total_reviews(reviews, fig_name='total-reviews.png'):
    fig, ax = plt.subplots(1)

    reviews_growth_by_year = reviews.groupby("year_written")["asin"].count()
    sns.barplot(reviews_growth_by_year.index, reviews_growth_by_year)
    ax.set_xticklabels(reviews_growth_by_year.index, rotation=30)
    ax.set_xlabel("Year")
    ax.set_ylabel("Num. of Reviews")
    ax.set_title("Amazon Total Reviews Year to Year")
    fig.savefig(fig_name)

def plot_helpfulness_of_reviews(reviews, fig_name='helpfulness-of-reviews.png'):
    fig, ax = plt.subplots(1)

    helpful_reviews = reviews.groupby("year_written")["percent_helpful"].mean()
    sns.barplot(helpful_reviews.index, helpful_reviews)
    ax.set_xticklabels(helpful_reviews.index, rotation=30)
    ax.set_xlabel("Year")
    ax.set_ylabel("Avg. Helpfulness of Reviews")
    ax.set_title("Amazon Review's Avg. Helpfulness Year to Year")
    fig.savefig(fig_name)

