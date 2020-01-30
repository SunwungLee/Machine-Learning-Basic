# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:41:12 2019

@author: taebe
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------
C=[[2,1], [1,2]]
C1=[[2,-1], [-1,2]]
#print(C)
A = np.linalg.cholesky(C)
A1 = np.linalg.cholesky(C1)
#print(A)
#print(A @ A.T)
print('eig= ', np.linalg.eig(C))
X = np.random.randn(10000,2)
Y = X @ A

Y1 = X @ A1
# ---------------------------------------------
# <6. Distribution of Projections>
theta = np.pi/3
u = [np.sin(theta), np.cos(theta)]
print('The vector: ', u)
print('Sum of squares: ', u[0]**2+u[1]**2)
print('Degree: ', theta*180/np.pi)
# ---------------------------------------------
yp = Y @ u
print(yp.shape)
print('Projected Variance: ', np.var(yp))
# ---------------------------------------------
nPoints = 50;
pVars = np.zeros(nPoints) # make 50x1 array
pVars1 = np.zeros(nPoints) # make 50x1 array
thRange = np.linspace(0, 2*np.pi, nPoints) # divided 0~2pi, 50 ea
for n in range(nPoints): # nPoints iterations
    theta = thRange[n] # divided theta
    u = [np.sin(theta), np.cos(theta)] # u vector is like a circle
    pVars[n] = np.var(Y @ u) # variance of projected matrix Y
    pVars1[n] = np.var(Y1 @ u) # variance of projected matrix Y
    
plt.title('3124232 / Sunwung Lee',  fontsize=14)
plt.plot(thRange*180/np.pi, pVars, label='C=[2, 1; 1, 2]')
plt.plot(thRange*180/np.pi, pVars1, label='C=[2, -1; -1, 2]')
plt.grid(True)

plt.xlabel('Direction', fontsize=14)
plt.ylabel('Variance of Projections', fontsize=14)
plt.legend()

#plt.savefig('6_Distribution_of_Projection-2')


# ---------------------------------------------