# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:29:10 2019

@author: taebe
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# Function Definition part
#
def gauss2D (x, m, C):
    Ci = np.linalg.inv(C) # inverse matrix of C
    dC = np.linalg.det(C1) # determinant of C-1
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
# Common variables
#
NumDataPerClass = 200

m1 = np.array([0, 3])
m2 = np.array([3, 2.5])
C1 = C2 = np.array([[2, 1], [1, 2]])
P1 = P2 = 0.5

A1 = np.linalg.cholesky(C1) # linear convert
A2 = np.linalg.cholesky(C2) # linear convert
X1 = np.random.randn(NumDataPerClass, 2) # operate random number (normal distribution)
Y1 = X1 @ A1 + m1 # linear convert
#plt.scatter(X1[:,0],X1[:,1], s=3,c='r')
#plt.scatter(Y1[:,0],Y1[:,1], s=3,c='b')
X2 = np.random.randn(NumDataPerClass, 2)
Y2 = X2 @ A2 + m2
#plt.scatter(X2[:,0],X2[:,1], s=3,c='r')
#plt.scatter(Y2[:,0],Y2[:,1], s=3,c='b')
#test = np.linspace(-5, 5, 200) # seperate 200 pieces from -5 to 5

# variables
C = np.array([[2, 1], [1, 2]])
Ci = np.linalg.inv(2*C)
uF = Ci @ (m2-m1)

yp1 = Y1 @ uF
yp2 = Y2 @ uF

# 3. Mahalanobis Distance
#
# Define a range over which to slide a threshold
#
pmin = np.min( np.array( (np.min(yp1), np.min(yp2) )))
pmax = np.max( np.array( (np.max(yp1), np.max(yp2) )))
print(pmin, pmax)

# Set up an array of thresholds
#
nRocPoints = 50
thRange = np.linspace(pmin, pmax, nRocPoints)
ROC = np.zeros( (nRocPoints, 2) )

# Compute True Positives and Flase positives at each threshold
#
for i in range(len(thRange)):
    thresh = thRange[i]
    TP = len(yp2[yp2 > thresh]) * 100 / len(yp2)
    FP = len(yp1[yp1 > thresh]) * 100 / len(yp1)
    ROC[i,:] = [TP, FP]
    
# Plot ROC curve
#
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(ROC[:,1], ROC[:,0], c='m')
ax.set_xlabel('False Positive')
ax.set_ylabel('True Positive')
ax.set_title('Receiver Operating Characteristics')
ax.grid(True)
#plt.savefig('rocCurve.png')