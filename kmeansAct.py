# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 21:30:11 2025

@author: NGATCH04
"""

import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

home_data = pd.read_csv('./results/matrices/act/M2_act.csv', usecols = ['code_etudiant','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','result'])
#home_data = pd.read_csv('./results/matrices/act/housing.csv', usecols = ['longitude', 'latitude', 'median_house_value'])
home_data.head()
#sns.scatterplot(data = home_data, x = 'code_etudiant', y = '1', hue = 'result')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(home_data[['code_etudiant', '2']], home_data[['result']], test_size=0.33, random_state=0)
X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)
kmeans = KMeans(n_clusters = 6, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)
sns.scatterplot(data = X_train, x = 'code_etudiant', y = '2', hue = kmeans.labels_)

#sns.boxplot(x = kmeans.labels_, y = y_train['result'])

#Meilleur choix du k
silhouette_score(X_train_norm, kmeans.labels_, metric='euclidean')
K = range(2, 10)
fits = []
score = []
i=0

for k in K:
    # train the model for current value of k on training data
    model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(X_train_norm)
    # append the model to fits
    fits.append(model)
    # Append the silhouette score to scores
    score.append(silhouette_score(X_train_norm, model.labels_, metric='euclidean'))
    sns.scatterplot(data = X_train, x = 'code_etudiant', y = '2', hue = fits[i].labels_)
    i=i+1

sns.lineplot(x = K, y = score)
X_train, X_test, y_train, y_test = train_test_split(home_data[['code_etudiant', '21']], home_data[['result']], test_size=0.33, random_state=0)
X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)
kmeans = KMeans(n_clusters = 7, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)
sns.scatterplot(data = X_train, x = 'code_etudiant', y = '21', hue = kmeans.labels_)
sns.boxplot(x = kmeans.labels_, y = y_train['result'])