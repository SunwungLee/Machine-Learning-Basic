# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:39:53 2019

@author: taebe
"""
import numpy as np
import matplotlib.pyplot as plt

# <1. Linear Algebra>
x = np.array([1, 2])
y = np.array([-2, 1])
a = np.dot(x,y)
print(a) # inner product

b = np.linalg.norm(x)
c = np.sqrt(x[0]**2+x[1]**2) # norm = x1^2 + x2^2
print(b, c) # prove

theta = np.arccos(np.dot(x, y) / (np.linalg.norm(x))*(np.linalg.norm(y)))
print(theta * 180 / np.pi) # degree --> orthogonal matrix

B = np.array([[3,2,1], [2,6,5], [1,5,9]], dtype=float)
print('-------------B is symmetric Matrix-------------')
print(B)
print(B - B.T) # it means Matrix B is symmetric matrix
z = np.random.rand(3)
v = B @ z
print('--------------------------------------')
print('matrix z: ', z)
print(v.shape)
print('matrix v: ',v)
print('--------------------------------------')
print('Quadratic form: ', z.T @ B @ z)
print('--------------------------------------')
print('Trace of Matrix B: ', np.trace(B))
print('Determinant of Matrix B: ', np.linalg.det(B))
print('--------------------------------------')
print('Inverse Matrix B @ B: ', np.linalg.inv(B) @ B)
print('--------------------------------------')
D, U = np.linalg.eig(B)
# RETURN
# w: eigenvalues
# v: normalized eigenvectors
print('--------------------------------------')
print(D)
print('--------------------------------------')
print(U)
print('--------------------------------------')
print(np.dot(U[:,0], U[:,1]))
print('--------------------------------------')
print(U @ U.T) # result = Identity matrix
# ---------------------------------------------

