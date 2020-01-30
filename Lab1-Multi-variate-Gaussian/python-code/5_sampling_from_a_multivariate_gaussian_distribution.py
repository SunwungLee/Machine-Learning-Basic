# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:41:11 2019

@author: taebe
"""
import numpy as np
import matplotlib.pyplot as plt
# <5. Sampling from a Multivariate Gaussian Distribution>
C=[[2,1], [1,2]]
print(C)
A = np.linalg.cholesky(C)
print(A) # lower triangular matrix
print(A @ A.T)
# ---------------------------------------------
X = np.random.randn(10000, 2)
Y = X @ A
print(A)
print(np.cov(Y[:,0],Y[:,1]))

# ---------------------------------------------
plt.scatter(Y[:,0], Y[:,1], s=3, c='m', label='matrix Y=X@A')
plt.scatter(X[:,0], X[:,1], s=3, c='c', label='matrix X')
plt.grid(True)
plt.title('Scatter of Isotropic and Corrleated Gaussian Densities')
plt.xlabel('31240232 / Sunwung Lee', fontsize=14)
plt.legend()
#plt.savefig('5_Sampling_from_a_Multivariate_Gaussian_Distribution')
# ---------------------------------------------