# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 22:24:28 2019

@author: taebe
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Function Definition part
#
def gauss2D (x, m, C):
    Ci = np.linalg.inv(C) # inverse matrix of C
    dC = np.linalg.det(Ci) # determinant of C-1
    num = np.exp(-0.5 * np.dot((x-m).T, np.dot(Ci, (x-m)))) 
    # gaussian distribution, Dot product of two arrays 그냥 곱셈.
    # making quadratic form (zT B z)
    den = 2 * np.pi * dC
    
    return num/den

def twoDGaussianPlot (nx, ny, m, C):
    x = np.linspace(-7, 7, nx) # 그래프 그리는 용도
    y = np.linspace(-7, 7, ny) # the same as x
    X, Y = np.meshgrid(x, y, indexing='ij') # make matrix
    
    # Return coordinate matrices from coordinate vectors.
    Z = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            xvec = np.array([X[i,j], Y[i,j]])
            Z[i,j] = gauss2D(xvec, m, C)
            
    return X, Y, Z

def posteriorPlot(nx, ny, m1, C1, m2, C2, P1, P2):
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    Z = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            xvec = np.array([X[i,j], Y[i,j]])
            num = P1*gauss2D(xvec, m1, C1)
            den = P1*gauss2D(xvec, m1, C1) + P2*gauss2D(xvec, m2, C2)
            Z[i,j] = num / den
            
    return X, Y, Z

# 1. Class Boundaries and Posterior Probabilities
#
# Common variables
#
NumDataPerClass = 200
# case 1
m1c1 = np.array([0, 3])
m2c1 = np.array([3, 2.5])
C1c1 = C2c1 = np.array([[2, 1], [1, 2]])
P1c1 = P2c1 = 0.5
# case 2
m1c2 = np.array([0, 3])
m2c2 = np.array([3, 2.5]) 
C1c2 = C2c2 = np.array([[2, 1], [1, 2]])
P1c2 = 0.7
P2c2 = 0.3
# case 3
m1c3 = np.array([0, 3])
m2c3 = np.array([3, 2.5]) 
C1c3 = np.array([[2, 0], [0, 2]])
C2c3 = np.array([[1.5, 0], [0, 1.5]])
P1c3 = P2c3 = 0.5

A1 = np.linalg.cholesky(C1c1) # linear convert
A2 = np.linalg.cholesky(C2c1) # linear convert
X1 = np.random.randn(NumDataPerClass, 2) # operate random number (normal distribution)
X2 = np.random.randn(NumDataPerClass, 2)

Y1 = X1 @ A1 + m1c1 # linear convert
Y2 = X2 @ A2 + m2c1
#plt.scatter(X2[:,0],X2[:,1], s=3,c='r')
#plt.scatter(Y2[:,0],Y2[:,1], s=3,c='r')
#test = np.linspace(-5, 5, 200) # seperate 200 pieces from -5 to 5

## case 1
##
A1 = np.linalg.cholesky(C1c1) # linear convert
A2 = np.linalg.cholesky(C2c1) # linear convert
Y1 = X1 @ A1 + m1c1 # linear convert
Y2 = X2 @ A2 + m2c1

plt.figure(figsize=(7,6))
plt.axis([-4, 8, -4, 8])
plt.grid(True)
plt.scatter(Y1[:,0],Y1[:,1], s=3, c='b')
plt.scatter(Y2[:,0],Y2[:,1], s=3, c='r')

XX1, YY1, ZZ1 = twoDGaussianPlot(NumDataPerClass, NumDataPerClass, m1c1, C1c1)
XX2, YY2, ZZ2 = twoDGaussianPlot(NumDataPerClass, NumDataPerClass, m2c1, C2c1)
plt.contour(XX1, YY1, ZZ1, 3)
plt.contour(XX2, YY2, ZZ2, 3)
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().legend(('m=[0, 3],C=[2,1;1,2],P=0.5','m=[3, 2.5],C=[2,1;1,2],P=0.5'))
plt.title('Case 1 - 31240232')
#plt.savefig('0_case1.png')
#
## case 2
##
A1 = np.linalg.cholesky(C1c2) # linear convert
A2 = np.linalg.cholesky(C2c2) # linear convert
Y1 = X1 @ A1 + m1c2 # linear convert
Y2 = X2 @ A2 + m2c2

plt.subplots(figsize=(7,6))
plt.axis([-4, 8, -4, 8])
plt.grid(True)
plt.scatter(Y1[:,0],Y1[:,1], s=3, c='b')
plt.scatter(Y2[:,0],Y2[:,1], s=3, c='r')
XX1, YY1, ZZ1 = twoDGaussianPlot(NumDataPerClass, NumDataPerClass, m1c2, C1c2)
XX2, YY2, ZZ2 = twoDGaussianPlot(NumDataPerClass, NumDataPerClass, m2c2, C2c2)
plt.contour(XX1, YY1, ZZ1, 3)
plt.contour(XX2, YY2, ZZ2, 3)
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().legend(('m=[0, 3],C=[2,1;1,2],P=0.7','m=[3, 2.5],C=[2,1;1,2],P=0.3'))
plt.title('Case 2 - 31240232')
#plt.savefig('0_case2.png')
#
## case 3
##
#A1 = np.linalg.cholesky(C1c3) # linear convert
#A2 = np.linalg.cholesky(C2c3) # linear convert
#Y1 = X1 @ A1 + m1c3 # linear convert
#Y2 = X2 @ A2 + m2c3
#
#plt.subplots(figsize=(7,6))
#plt.axis([-4, 8, -4, 8])
#plt.grid(True)
#plt.scatter(Y1[:,0],Y1[:,1], s=3, c='b')
#plt.scatter(Y2[:,0],Y2[:,1], s=3, c='r')
#XX1, YY1, ZZ1 = twoDGaussianPlot(NumDataPerClass, NumDataPerClass, m1c3, C1c3)
#XX2, YY2, ZZ2 = twoDGaussianPlot(NumDataPerClass, NumDataPerClass, m2c3, C2c3)
#plt.contour(XX1, YY1, ZZ1, 3)
#plt.contour(XX2, YY2, ZZ2, 3)
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.gca().legend(('m=[0, 3],C=[2,0;0,2],P=0.5','m=[3, 2.5],C=[1.5,0;0,1.5],P=0.5'))
#plt.title('Case 3 - 31240232')
#plt.savefig('0_case3.png')

#####################

plt.figure(figsize=(7,6))
plt.axis([-5, 5, -5, 5])
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
#
#X, Y, Z = posteriorPlot(NumDataPerClass, NumDataPerClass, m1c1, C1c1, m2c1, C2c1, P1c1, P2c1)
#plt.title('posteriorPlot-case1 - 31240232')
#plt.contour(X, Y, Z, 4)

#X, Y, Z = posteriorPlot(NumDataPerClass, NumDataPerClass, m1c2, C1c2, m2c2, C2c2, P1c2, P2c2)
#plt.title('posteriorPlot-case2 - 31240232')
#plt.contour(X, Y, Z, 4)

X, Y, Z = posteriorPlot(NumDataPerClass, NumDataPerClass, m1c3, C1c3, m2c3, C2c3, P1c3, P2c3)
plt.title('posteriorPlot-case3 - 31240232')
plt.contour(X, Y, Z, 4)

#plt.savefig('pos-case1.png')

#P1 = 0.7
#P2 = 0.5
#X, Y, Z = posteriorPlot(NumDataPerClass, NumDataPerClass, m1, C1, m2, C2, P1, P2)
#print('X: ', X)
#print('Y: ', Y)
#plt.contour(X, Y, Z, 4)






