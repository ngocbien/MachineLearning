# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:19:35 2017

@author: NgocBien
"""

import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import pandas as pd;
from sklearn import decomposition
#from sklearn import datasets
from sklearn.preprocessing import StandardScaler as scaler
import os
os.chdir("C:/Users/NgocBien/Desktop/MachineLearningProjet/MachineLearning/TPML/TPML")
data2 = pd.read_csv('./crime.csv', sep=';');
X2=data2.ix[:,1:7].values
labels2=data2.ix[:,0].values
pca = decomposition.PCA(n_components=3)
#ces codes en base nous permettent de savoir combien d'infos qu'on garde 
#quand on fait de ACP.
X2_norm=scaler().fit_transform(X2)
pca.fit(X2_norm)
print(pca.singular_values_)
print(pca.explained_variance_ratio_)
# On recupere les coordonnees de PCA sur 3 axes et on fait la projection 
#sur les 2 premieres axes.
X2_pca=pca.fit_transform(X2)
import matplotlib
plt.scatter(X2_pca[:, 0], X2_pca[:, 1])
for label, x, y in zip(labels2, X2_pca[:, 0], X2_pca[:, 1]):
   plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
plt.show()