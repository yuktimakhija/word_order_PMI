import pickle
from pmi_features import preprocess_cluster
import os
import json
import numpy as np
import plotly.express as px
from sklearn import metrics

k_clusters = [16,24,32,48,64,128]
scores = {
    # 'k':[],
    'inertia':[],
    'adjusted_mutual_info_score':[],
    'adjusted_rand_index':[],
    'silhouette_score':[],
    'calinski_harabasz_score':[]
    }
sample_size = 10000
tags_sent = preprocess_cluster('all_data.dep')[-1]
tags = []
for sent in tags_sent:
    tags += sent

for k in k_clusters:
    kmeans = pickle.load(open(f'{k}/kmeans_.pkl', 'rb'))
    # x = pickle.load(open(f'{k}/word_embeddings_after_training','rb'))
    X = np.load(f'{k}/word_embs.npy')
    labels = kmeans.labels_
    # scores['k'].append(k)
    print(k)
    scores['inertia'].append(kmeans.inertia_)
    print('AMI')
    scores['adjusted_mutual_info_score'].append(metrics.adjusted_mutual_info_score(labels, tags))
    print('ARI')
    scores['adjusted_rand_index'].append(metrics.adjusted_rand_score(labels, tags))
    print('sil')
    scores['silhouette_score'].append(metrics.silhouette_score(X, labels, sample_size=sample_size))
    print('CH')
    scores['calinski_harabasz_score'].append(metrics.calinski_harabasz_score(X[:sample_size,:], labels[:sample_size]))
    # for sent in tags:

json.dump(scores, open('scores.json', 'w'))
px.line(scores).write_html('plots.html')
