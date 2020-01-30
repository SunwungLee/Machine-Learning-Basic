# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:41:11 2019

@author: taebe
"""
import numpy as np
import matplotlib.pyplot as plt
# <4. Bivariate Gaussian Distribution>
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
# ---------------------------------------------
nx, ny = 50, 40
plt.figure(figsize=(6,6))

#m1 = np.array([0, 2])
m1 = np.array([2, 0])
#m1 = np.array([-2, -2])
#C1 = np.array([[2,1], [1,2]], np.float32)
C1 = np.array([[1.5,-1], [-1,1.5]], np.float32)
#C1 = np.array([[2,0], [0,2]], np.float32)
Xp, Yp, Zp = twoDGaussianPlot(nx, ny, m1, C1)
#plt.contour(Xp, Yp, Zp, 3)
plt.contour(Xp, Yp, Zp, 15)
plt.grid(True)

plt.title('31240232/Sunwung Lee', fontsize=14)
plt.xlabel('m=[2,0], C=[2,-1; -1,2]', fontsize=14)
#plt.savefig('4_Bivariate_Gaussian_Distribution-2')
# ---------------------------------------------