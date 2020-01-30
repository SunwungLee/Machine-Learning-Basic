# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:24:14 2019

@author: taebe
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

from sklearn import svm
from sklearn.model_selection import KFold
# Gaussian 
#
def gaussian(x, u, sigma):
    return (np.exp(-0.5 * np.linalg.norm(x-u) / sigma))

diabetes = datasets.load_diabetes()
X = diabetes.data # input data
y = diabetes.target # target
# -->Normalize each feature of the input data to have a 
# mean of 0 and standard deviation of 1
#
Xnorm = (X-np.min(X)) / (np.max(X)-np.min(X)) # good

# Space for design matrix
#
M = 200
S = 10
# K-folds cross validation
#
kf = KFold(n_splits=S)
error_tr = list()
error_te = list()
lamda = list()
ytt = list()
for tr_idx, te_idx in kf.split(X):
    Xntr, Xnte = Xnorm[tr_idx], Xnorm[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]
    
    N, p = Xntr.shape
    Nt = Xnte.shape[0]
    Utr = np.zeros((N,M)) # row, 200
    Ute = np.zeros((Nt,M)) # row, 200
    
    Ck = KMeans(n_clusters=M, random_state=0).fit(Xntr) # Why divided into M ?
    C = Ck.cluster_centers_ # 200x10
    
    sigtmp = np.zeros(M) # good
    for i in range(M):
        x1 = Xntr[np.floor(np.random.rand()*N).astype(int), :] 
        x2 = Xntr[np.floor(np.random.rand()*N).astype(int), :]
        sigtmp[i] = np.linalg.norm(x1-x2) # sqrt(sum((x1-x2)^2))    
    sigma = np.mean(sigtmp)

    for i in range(N):
        for j in range(M):
            Utr[i,j] = gaussian(Xntr[i,:], C[j,:], sigma) # good
            
    for i in range(Nt):
        for j in range(M):
            Ute[i,j] = gaussian(Xnte[i,:], C[j,:], sigma) # good
            
    l = np.linalg.inv(Utr.T @ Utr) @ Utr.T @ ytr

    yh = Utr @ l
    yt = Ute @ l

    error_tr.append(np.sum(np.square(ytr-yh)))
    error_te.append(np.sum(np.square(yte-yt)))
    lamda.append(l)
    ytt.append(yt)

#print("min idx: ", np.argmin(error_te), "lamda: ", lamda[np.argmin(error_te)])



# Predicted values on training data
#
#fig = plt.figure(figsize=(9,6))
#plt.scatter(ytr, yh, c='m', s=5)
#plt.scatter(yte, yt, c='b', s=5)
#plt.grid(True)
#plt.title("31240232-Distribution of training/test set result", fontsize=14)
#plt.xlabel("True Target", fontsize=12)
#plt.ylabel("Prediction", fontsize=12)
#plt.gca().legend(('training','test'))
##plt.savefig('1-min_lamda_result.png')
#
#fig = plt.figure(figsize=(9,6))
#plt.boxplot(ytt)
#plt.grid(True)
#plt.title("31240232-Distributions of test set results", fontsize=14)
#plt.xlabel("#cross-validation", fontsize=12)
#plt.ylabel("test set results", fontsize=12)
#plt.savefig('1-boxplot_sbs.png')

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

svclassifier = SVC(kernel='rbf')
svclassifier.fit(Utr, yh.astype('int'))

y_pred = svclassifier.predict(Ute)
print(confusion_matrix(yte, y_pred))
print(classification_report(yte, y_pred))
