# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:05:32 2019

@author: taebe
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load data, inspect and do exploratory plots
# diabetes means 당뇨병
#
diabetes = datasets.load_diabetes() # bring the dataset from sklearn

Y = diabetes.data       # data = matrix Y
f = diabetes.target     # 측정한 결과값
NumData, NumFeatures = Y.shape
print(NumData, NumFeatures)

print(f.shape)

diabetes = datasets.load_diabetes()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
ax[0].hist(f, bins=40)
ax[0].set_title("Distribution of Target", fontsize=14)
ax[1].scatter(Y[:,6], Y[:,7], c='m', s=3) # 단순히 데이터 디스플레이?
ax[1].set_title("Scatter of Two Inputs", fontsize=14)


# Linear regression using sklearn
#
lin = LinearRegression()    # linear regression implement
# Train the model using the whole data and target
lin.fit(Y, f)               # lin --> make linear regression model 
fh1 = lin.predict(Y)        # make 

# Pseudo-inverse solution to linear regression
#
a = np.linalg.inv(Y.T @ Y) @ Y.T @ f # matrix a is coefficient matrix
fh2 = Y @ a                 # linear predictor = data * coeff (except error)

# Plot predictions to check if they look the same!
#
fig = plt.figure()
plt.subplots(figsize=(7,5))
plt.scatter(f, fh1, c='c', s=3) # target, data*coeff
plt.grid(True)
plt.ylabel('prediction')
plt.xlabel('target')
plt.title("31240232-Sklearn", fontsize=14)
plt.savefig('1-sklearn.png')

fig = plt.figure()
plt.subplots(figsize=(7,5))
plt.scatter(f, fh2, c='m', s=3)
plt.grid(True)
plt.ylabel('prediction')
plt.xlabel('target')
plt.title("31240232-Pseudoinverse", fontsize=14)
plt.savefig('1-Pseudo-inverse.png')

#fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
#ax[0].scatter(f, fh1, c='c', s=3) # target, data*coeff
#ax[0].grid(True)
#ax[0].set_title("Sklearn", fontsize=14)
#
#ax[1].scatter(f, fh2, c='m', s=3)
#ax[1].grid(True)
#ax[1].set_title("Pseudoinverse", fontsize=14)
## the same tendency result. but, the values of Y-axis(=fh1,2) are different

# Tikhanov Regularizer(Ridge Regularization)
#
gamma = 0.7
aR = np.linalg.inv(Y.T @ Y + gamma*np.identity(NumFeatures)) @ Y.T @ f
## what is the role of gamma?
fig = plt.figure()
plt.subplots(figsize=(7,5))
plt.bar(np.arange(len(a)), a)
plt.title('31240232-Pseudo-inverse solution', fontsize=14)
plt.grid(True)
plt.ylim(np.min(a), np.max(a))
plt.ylabel('value of coeff')
plt.xlabel('number of coeff')
#plt.savefig('2-Pseudo-inverse_solution.png')

fig = plt.figure()
plt.subplots(figsize=(7,5))
plt.bar(np.arange(len(aR)), aR)
plt.title('31240232-Ridge Regularized solution', fontsize=14)
plt.grid(True)
plt.ylabel('value of coeff')
plt.xlabel('number of coeff')
plt.ylim(np.min(a), np.max(a))
#plt.savefig('2-Ridge_regularized.png')

#fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
#ax[0].bar(np.arange(len(a)), a)
#ax[0].set_title('31240232-Pseudo-inverse solution', fontsize=14)
#ax[0].grid(True)
#ax[0].set_ylim(np.min(a), np.max(a))
#
#ax[1].bar(np.arange(len(aR)), aR)
#ax[1].set_title('31240232-Regularized solution', fontsize=14)
#ax[1].grid(True)
#ax[1].set_ylim(np.min(a), np.max(a))
## shrinkage of a --> a가 작을수록 영향 x, a가 클수록 coeff 0으로 수렴
fig = plt.figure()
plt.subplots(figsize=(7,5))
plt.hist(fh2, bins=40)
plt.hist(Y@aR, bins=40)
plt.title('31240232-Distributions', fontsize=14)
plt.ylabel('number of coeff')
plt.xlabel('the value of fh2')
plt.gca().legend(('psuedo-inv','ridge'))
#plt.savefig('3-distributions.png')

fig = plt.figure()
plt.subplots(figsize=(7,5))
plt.scatter(f, fh2, c='c', s=3)
plt.scatter(f, Y@aR, c='m', s=3)
plt.title('31240232-predictions', fontsize=14)
plt.ylabel('predicted target')
plt.xlabel('target')
plt.grid(True)
plt.gca().legend(('psuedo-inv','ridge'))
#plt.savefig('3-predictions.png')
#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,4))
#ax.hist(fh2, bins=40)
#ax.hist(Y@aR, bins=40)
#fig, ax = plt.subplots(nrows=1, ncols=1)
#ax.scatter(f, fh2, c='c', s=3) # target, data*coeff
#ax.scatter(f, Y@aR, c='m', s=3)
#ax.grid(True)


# Sparsity inducing (lasso) regularizer
#
from sklearn.linear_model import Lasso
ll = Lasso(alpha=0.2)
ll.fit(Y, f)
yh_lasso = ll.predict(Y)

fig = plt.figure()
plt.subplots(figsize=(7,5))
plt.bar(np.arange(len(a)), a)
plt.title('31240232-Pseudo-inverse solution', fontsize=14)
plt.grid(True)
plt.ylim(np.min(a), np.max(a))
plt.ylabel('value of coeff')
plt.xlabel('number of coeff')
#plt.savefig('4-Pseudo-inverse_solution.png')

fig = plt.figure()
plt.subplots(figsize=(7,5))
plt.bar(np.arange(len(ll.coef_)), ll.coef_)
plt.title('31240232-Lasso solution', fontsize=14)
plt.grid(True)
plt.ylabel('value of coeff')
plt.xlabel('number of coeff')
plt.ylim(np.min(a), np.max(a))
#plt.savefig('4-Lasso_solution.png')

#fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
#ax[0].bar(np.arange(len(a)), a)
#ax[0].set_title('Pseudo-inverse solution', fontsize=14)
#ax[0].grid(True)
#ax[0].set_ylim(np.min(a), np.max(a))
#ax[1].bar(np.arange(len(ll.coef_)), ll.coef_)
#ax[1].set_title('Lasso solution', fontsize=14)
#ax[1].grid(True)
#ax[1].set_ylim(np.min(a), np.max(a))

# Lasso Regularization path on a synthetic example
#
from sklearn import linear_model

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

print("Computing regularization path using the LARS ...")
_, _, coefs = linear_model.lars_path(X, y, method='lasso', verbose=True)
xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]
fig = plt.figure()
plt.subplots(figsize=(10,5))
plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('31240232-LASSO Path', fontsize=14)
plt.axis('tight')
#plt.savefig('5-lasso_path.png')