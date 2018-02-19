# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 08:45:29 2017

@author: NgocBien
"""
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import pandas as pd;
from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler as scaler
import os
os.chdir("C:/Users/NgocBien/Desktop/MachineLearningProjet/MachineLearning/TPML/TPML")
data = pd.read_csv('./villes.csv', sep=';');
X = data.ix[:, 1:13].values
labels = data.ix[:, 0].values
#how to center X?
X_norm=scaler().fit_transform(X)
pca = decomposition.PCA(n_components=3)
pca.fit(X_norm)
print(pca.singular_values_)
print(pca.explained_variance_ratio_)
#Creer les projections sur les 3 principales axes:
X_pca=pca.fit_transform(X)
import matplotlib
plt.scatter(X_pca[:, 0], X_pca[:, 1])
for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
   plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
plt.show()

#Nous faisons la methode k-means sur les donnees pour avoir 3 clustering:
from sklearn.cluster import KMeans;
k_means=KMeans(n_clusters=3,random_state=0);
cluster=k_means.predict(X_norm)
#Nous faisons ici la projection avec les differentes couleurs des 
#clusters sur les differentes axes:
colors = ['red','yellow','blue','pink']
plt.scatter(X_pca[:, 0], X_pca[:, 1], c= cluster, cmap=matplotlib.colors.ListedColormap(colors))
for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
  plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
plt.show()

