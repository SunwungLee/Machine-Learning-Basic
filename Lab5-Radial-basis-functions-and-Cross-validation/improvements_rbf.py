# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:08:47 2019

@author: taebe
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

from sklearn.svm import SVR

diabetes = datasets.load_diabetes()
X = diabetes.data # input data
y = diabetes.target # target
# -->Normalize each feature of the input data to have a 
# mean of 0 and standard deviation of 1
#
Xnorm = (X-np.min(X)) / (np.max(X)-np.min(X))

# Gaussian 
#
def gaussian(x, u, sigma):
    return (np.exp(-0.5 * np.linalg.norm(x-u) / sigma))

#N, p = X.shape
N, p = Xnorm.shape
print(N, p) # N: row, p: col

# Space for design matrix
#
M = 200
U = np.zeros((N,M)) # row, 200

# Basis function locations at random
# --> K-means clustering (K=M) and set C = centre of clustering
#
Ck = KMeans(n_clusters=M, random_state=0).fit(Xnorm)
C = Ck.cluster_centers_
tt = Ck.predict(Xnorm)
plt.scatter(Xnorm[:,2], Xnorm[:,3], c=tt)
# Basis function range as distance between two random data
# x1, x2 means random column vector of X matrix
# --> change it to be the average of several pairwise distances.
#
sigtmp = np.zeros(M)
for i in range(M):
    x1 = Xnorm[np.floor(np.random.rand()*N).astype(int), :] 
    x2 = Xnorm[np.floor(np.random.rand()*N).astype(int), :]
    sigtmp[i] = np.linalg.norm(x1-x2) # sqrt(sum((x1-x2)^2))    
sigma = np.mean(sigtmp)


# --> split the data into training and test sets, estimate the model
# and note the test set performance
#
a = np.split(Xnorm, 13) # 이 함수는 나누어떨어지게만 split 가능
Xtr = a[0:11]
Xte = a[12]
np.asarray(Xtr)
np.asarray(Xte)

# Construct the design matrix
#
for i in range(N):
    for j in range(M):
        U[i,j] = gaussian(Xnorm[i,:], C[j,:], sigma) #
#        U[i,j] = gaussian(Xtr[i,:], C[j,:], sigma) #
        
# Pseudo inverse solution for linear part
#
l = np.linalg.inv(U.T @ U) @ U.T @ y

# Predicted values on training data
#
yh = U @ l

fig, ax = plt.subplots(figsize=(3,3))
ax.scatter(y, yh, c='m', s=3)
ax.grid(True)
ax.set_title("Training Set", fontsize=14)
ax.set_xlabel("True Target", fontsize=12)
ax.set_ylabel("Prediction", fontsize=12)

# K-folds cross validation
#
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
a=0
for x1, x2 in kf.split(X):
    a=a+1
    print("train: ", x1, "test: ", x2)
    xtr, xte = X[x1], X[x2]
    ytr, yte = y[x1], y[x2]
    print("--------------------", a)
    

## sklearn RBF kernel example
##
#clf = SVR(gamma='scale', C=1000000)
#clf.fit(X, y)
#clf.fit(X, y).predict(X)
#ax.scatter(y, clf.fit(X, y).predict(X), c='b', s=3)