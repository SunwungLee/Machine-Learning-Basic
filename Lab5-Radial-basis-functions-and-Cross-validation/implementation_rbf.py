# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:19:04 2019

@author: taebe
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LinearRegression

diabetes = datasets.load_diabetes()
X = diabetes.data # input data 
y = diabetes.target # target

# Gaussian (PHI function)
#
def gaussian(x, u, sigma):
    return (np.exp(-0.5 * np.linalg.norm(x-u) / sigma))

#N, p = X.shape
N, p = X.shape
print(N, p) # N: row, p: col

# Space for design matrix
#
M = 200 # it must be smaller than N
U = np.zeros((N,M)) # row, 200

# Basis function locations at random
# mj, C = centre
#
C = np.random.randn(M, p) # 200, col

# Basis function range as distance between two random data
# x1, x2 means random column vector of X matrix
# = Radial
#
x1 = X[np.floor(np.random.rand()*N).astype(int), :] 
x2 = X[np.floor(np.random.rand()*N).astype(int), :]
sigma = np.linalg.norm(x1-x2) # sqrt(sum((x1-x2)^2))

# Construct the design matrix
#
for i in range(N):
    for j in range(M):
        U[i,j] = gaussian(X[i,:], C[j,:], sigma) # Y = NxM

# Pseudo inverse solution for linear part
# lamda = min(|Y@lamda - f|^2)
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


