# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:43:06 2019

@author: taebe
"""

#%mathplot inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
sol = pd.read_excel("Husskonen_Solubility_Features.xlsx", verbose=False)
#print(sol.shape)

colnames = sol.columns
#print(colnames)

f = sol["LogS.M."].values
fig, ax = plt.subplots(figsize=(4,4))
ax.hist(f, bins=40, facecolor='m')
ax.set_title("Histogram of Log Solubility", fontsize=14)
ax.grid(True)

Y = sol[colnames[5:len(colnames)]]
N, p = Y.shape

print(Y.shape)
print(f.shape)

# Split data into training and test sets
#
from sklearn.model_selection import train_test_split
Y_train, Y_test, f_train, f_test = train_test_split(Y, f, test_size=0.3)

# Regularized regression
#
gamma = 2.3
sa = np.linalg.inv(Y_train.T @ Y_train + gamma*np.identity(p)) @ Y_train.T @ f_train
a = sa.to_numpy()
fh_train = Y_train @ a
fh_test = Y_test @ a

## Plot training and test predictions
##
#fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
#ax[0].scatter(f_train, fh_train, c='m', s=3)
#ax[0].grid(True)
#ax[0].set_title("Training Data", fontsize=14)
#
#ax[1].scatter(f_test, fh_test, c='m', s=3)
#ax[1].grid(True)
#ax[1].set_title("Test Data", fontsize=14)

# Over to you for implementing Lasso
#
from sklearn.linear_model import Lasso
ll = Lasso(alpha=2.3)
ll.fit(Y_train, f_train)
fl_train = ll.predict(Y_train)
fl_test = ll.predict(Y_test)

## Plot training and test predictions
##
#fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
#ax[0].scatter(f_train, fl_train, c='m', s=3)
#ax[0].grid(True)
#ax[0].set_title("Training Data", fontsize=14)
#
#ax[1].scatter(f_test, fl_test, c='m', s=3)
#ax[1].grid(True)
#ax[1].set_title("Test Data", fontsize=14)

fig = plt.figure()
plt.subplots(figsize=(7,5))
plt.scatter(f_test, fh_test, c='m', s=3)
plt.title('31240232-Test data', fontsize=14)
plt.ylabel('predicted target')
plt.xlabel('target')
plt.grid(True)
plt.savefig('6-L2-test.png')

fig = plt.figure()
plt.subplots(figsize=(7,5))
plt.scatter(f_train, fh_train, c='c', s=3)
plt.title('31240232-Training data', fontsize=14)
plt.ylabel('predicted target')
plt.xlabel('target')
plt.grid(True)
plt.savefig('6-L2-training.png')

fig = plt.figure()
plt.subplots(figsize=(7,5))
plt.scatter(f_test, fl_test, c='m', s=3)
plt.title('31240232-Test data', fontsize=14)
plt.ylabel('predicted target')
plt.xlabel('target')
plt.grid(True)
plt.savefig('7-L1-test.png')

fig = plt.figure()
plt.subplots(figsize=(7,5))
plt.scatter(f_train, fl_train, c='c', s=3)
plt.title('31240232-Training data', fontsize=14)
plt.ylabel('predicted target')
plt.xlabel('target')
plt.grid(True)
plt.savefig('7-L1-training.png')


###############
fig = plt.figure()
plt.title('31240232-Distributions', fontsize=14)
plt.hist(f_test, bins=40, facecolor='k')
plt.hist(fh_test, bins=40, facecolor='m')
plt.grid(True)
plt.ylabel('number of coeff')
plt.xlabel('predictions by Ridge')
plt.gca().legend(('target','ridge'))
plt.savefig('6-predictions.png')

fig = plt.figure()
plt.title('31240232-Distributions', fontsize=14)
plt.hist(f_test, bins=40, facecolor='k')
plt.hist(fl_test, bins=40, facecolor='b')
plt.grid(True)
plt.ylabel('number of coeff')
plt.xlabel('predictions by Lasso')
plt.gca().legend(('target','lasso'))
plt.savefig('7-predictions.png')