# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:27:56 2019

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

X2 = np.random.randn(NumDataPerClass, 2)
Y2 = X2 @ A2 + m2

#plt.scatter(Y1[:,0],Y1[:,1], s=3,c='b')
#plt.scatter(Y2[:,0],Y2[:,1], s=3,c='b')

# 2. Fisher LDA and ROC Curve
#
C = np.array([[2, 1], [1, 2]])
Ci = np.linalg.inv(C1+C2) # because C1=C2 --> 2*C
uF = Ci @ (m2-m1) # wf = C^-1(m2-m1), Fisher LDA direction 
#print(uF) # ax+by+c=0 --> y = uFx - c/b

x = np.linspace(-5, 5, NumDataPerClass)
y = np.linspace(-5, 5, NumDataPerClass)
xx = np.linspace(-5,7, NumDataPerClass)
yy = (uF[1]/uF[0])*x
#yy1 = (uF[1]/uF[0])*x + 3
XX1, YY1, ZZ1 = twoDGaussianPlot(NumDataPerClass, NumDataPerClass, m1, C1)
XX2, YY2, ZZ2 = twoDGaussianPlot(NumDataPerClass, NumDataPerClass, m2, C2)

plt.figure(figsize=(7,6))
plt.axis([-4, 7, -4, 7])
plt.grid(True)
plt.plot(xx,yy, 'k')
#plt.plot(x,yy1, 'm')
plt.scatter(Y1[:,0], Y1[:,1], s=3, c='r')
plt.scatter(Y2[:,0], Y2[:,1], s=3, c='b')
plt.contour(XX1,YY1,ZZ1, 3)
plt.contour(XX2,YY2,ZZ2, 3)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('31240232/Sunwung Lee')
plt.gca().legend(('Fisher LDA','m=[0, 3],C=[2,1;1,2]','m=[3, 2.5],C=[2,1;1,2]'))

# Save figure
#
#plt.savefig('1_Random_Data_scatter-contour.png')
#plt.savefig('2_Fisher_LDA.png')

# Project the data onto the Fisher LDA by using dot product
#
yp1 = Y1 @ uF
yp2 = Y2 @ uF

plt.figure(figsize=(7,6))
matplotlib.rcParams.update({'font.size': 12})
plt.hist(yp1, bins=40)
plt.hist(yp2, bins=40)
plt.title('project the data onto the Fisher LDA - 31240232')
plt.gca().legend(('m=[0, 3],C=[2,1;1,2]','m=[3, 2.5],C=[2,1;1,2]'))
#plt.savefig('3_histogramprojections.png')

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
# Define a range over which to slide a threshold
#
yp1 = Y1 @ uF
yp2 = Y2 @ uF
pmin = np.min( np.array( (np.min(yp1), np.min(yp2) )))
pmax = np.max( np.array( (np.max(yp1), np.max(yp2) )))
#print('pmin: ', pmin, '/ pmax: ', pmax)

# Set up an array of thresholds
#
nRocPoints = 50
thRange = np.linspace(pmin, pmax, nRocPoints) # create space
ROC = np.zeros( (nRocPoints, 2) ) # make nRocPoints X 2 matrix
RT = np.zeros((nRocPoints, 2))
# Compute True Positives and False positives at each threshold
#
for i in range(len(thRange)): # iterate 50 times
    thresh = thRange[i] # threshold pmin ~ pmax interval : 50
#    TP = len(yp2[yp2 > thresh]) / len(yp2) # it sees Y2 is positive
#    FP = len(yp1[yp1 > thresh]) / len(yp1) 
    TP = len(yp2[yp2 > thresh]) * 100 / len(yp2) # it sees Y2 is positive
    FP = len(yp1[yp1 > thresh]) * 100 / len(yp1) 
    # so, if Y1 is bigger than threshold, it would become False Positive
    # why did *100 ? -> express percentage because I want to analyse easy way
    ROC[i,:] = [TP, FP] # store percentage
    tptn = len(yp2[yp2 > thresh])+len(yp1[yp1 < thresh])
    ac = (tptn/(len(yp2)+len(yp1))) * 100
    RT[i,:] = [ac, thresh]
    
#print('accuracy: ', RT)
print('max : ', np.max(RT[:,0]), '/ max threshold: ', thRange[np.argmax(RT[:,0])])
    
# Plot ROC curve
#
AUC = -np.trapz(ROC[:,0], x=ROC[:,1])
#print('AUC : ', AUC)

fig, ax = plt.subplots(figsize=(7,6))
ax.plot(ROC[:,1], ROC[:,0], c='m') # because axis x is FP and y is TP
ax.set_xlabel('False Positive rate')
ax.set_ylabel('True Positive rate')
ax.set_title('Receiver Operating Characteristics-31240232')
ax.grid(True)
#plt.savefig('4_rocCurve.png')

# AUC : Area Under the Curve -> represent accuracy
# 곡선이 굽어질수록 더 정확한 모델이다.

# Plot the ROC Curve (on the same scale) for
# - A random direction(instead of Fisher LDA)
# - the direction connecting m1, m2
# compare AUC 
uF_rdm = np.random.rand(2, )
uF_m = np.array(((m1[1]-m2[1]), (m1[0]-m2[0])))

yp1_r = Y1 @ uF_rdm
yp2_r = Y2 @ uF_rdm

yp1_m = Y1 @ uF_m
yp2_m = Y2 @ uF_m
#print('uF_rdm: ', uF_rdm, 'uF: ', uF, 'uF_m: ', uF_m)

pmin_r = np.min( np.array( (np.min(yp1_r), np.min(yp2_r) )))
pmax_r = np.max( np.array( (np.max(yp1_r), np.max(yp2_r) )))

pmin_m = np.min( np.array( (np.min(yp1_m), np.min(yp2_m) )))
pmax_m = np.max( np.array( (np.max(yp1_m), np.max(yp2_m) )))

thRange_r = np.linspace(pmin_r, pmax_r, nRocPoints) # create space
ROC_r = np.zeros( (nRocPoints, 2) ) # make nRocPoints X 2 matrix

thRange_m = np.linspace(pmin_m, pmax_m, nRocPoints) # create space
ROC_m = np.zeros( (nRocPoints, 2) ) # make nRocPoints X 2 matrix

for i in range(len(thRange)): # iterate 50 times
    thresh_r = thRange_r[i] # threshold pmin ~ pmax interval : 50
    thresh_m = thRange_m[i] # threshold pmin ~ pmax interval : 50

    TP_r = len(yp2_r[yp2_r > thresh_r]) * 100 / len(yp2_r) # it sees Y2 is positive
    FP_r = len(yp1_r[yp1_r > thresh_r]) * 100 / len(yp1_r) 
    
    TP_m = len(yp2_m[yp2_m > thresh_m]) * 100 / len(yp2_m) # it sees Y2 is positive
    FP_m = len(yp1_m[yp1_m > thresh_m]) * 100 / len(yp1_m) 
    
    ROC_r[i,:] = [TP_r, FP_r] # random
    ROC_m[i,:] = [TP_m, FP_m] # means

ax.plot(ROC_r[:,1], ROC_r[:,0], c='b')
ax.plot(ROC_m[:,1], ROC_m[:,0], c='k') 
#
plt.gca().legend(('Fisher LDA', 'Random Number', 'Connecting means'))
#plt.savefig('5_compare_rocCurve.png')






