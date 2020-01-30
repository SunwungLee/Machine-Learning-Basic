# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 01:56:01 2019

@author: taebe
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Euclidean Distance Caculator
def EuclDist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
# Generate samples by using Gaussian probability density
def genGaussianSamples(N, m, C):
    A = np.linalg.cholesky(C)   # what is this?
    U = np.random.randn(N, 2)   # what is this?
    
    return(U@A.T + m)           
    # expain the formula of generate Gaussian Samples

def gauss2D (x, m, C):
    Ci = np.linalg.inv(C) # inverse matrix of C
    dC = np.linalg.det(Ci)
    num = np.exp(-0.5 * np.dot((x-m).T, np.dot(Ci, (x-m)))) 
    # gaussian distribution, Dot product of two arrays 그냥 곱셈.
    # making quadratic form (zT B z)
    den = 2 * np.pi * dC
    
    return num/den

def twoDGaussianPlot (nx, ny, m, C):
    x = np.linspace(-6, 7, nx) # 그래프 그리는 용도
    y = np.linspace(-5, 8, ny) # the same as x
    X, Y = np.meshgrid(x, y, indexing='ij') # make matrix
    
    # Return coordinate matrices from coordinate vectors.
    Z = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            xvec = np.array([X[i,j], Y[i,j]])
            Z[i,j] = gauss2D(xvec, m, C)
            
    return X, Y, Z

# In[]: prepare the Gaussian samples
    
    
# Define three means, arbitary
#
Means = np.array([[0, 3], [3, 0], [4, 4]])

# Define three covariance matrices ensuring
# they are positive definite
#
from sklearn.datasets import make_spd_matrix
# 왜 3,2,2 인지 -> 3개의 2x2 Covariance Matrix 생성
CovMatrices = np.zeros((3,2,2)) 
for j in range(3):
    CovMatrices[j,:,:] = make_spd_matrix(2) 
    # 이게 무슨 function인
    # The random symmetric, positive-definite matrix
    
# Priors, 3개의 random weight 생성
# and normalization, sum of weights is 1
#
w = np.random.rand(3)
w = w / np.sum(w)

# How many data in each component (1000 in total)
#
nData = np.floor(w * 1000).astype(int)

# Draw samples from each component
#
X0 = genGaussianSamples(nData[0], Means[0,:], CovMatrices[0,:,:])
X1 = genGaussianSamples(nData[1], Means[1,:], CovMatrices[0,:,:])
X2 = genGaussianSamples(nData[2], Means[2,:], CovMatrices[0,:,:])
#X1 = genGaussianSamples(nData[1], Means[1,:], CovMatrices[1,:,:])
#X2 = genGaussianSamples(nData[2], Means[2,:], CovMatrices[2,:,:])

# Plot scatter of samples
#
fig = plt.figure()
plt.subplots(figsize=(7,5))
plt.xlim([-6, 9])
plt.ylim([-5, 8])

plt.scatter(X0[:,0], X0[:,1], c='k', s=1)
plt.scatter(X1[:,0], X1[:,1], c='b', s=1)
plt.scatter(X2[:,0], X2[:,1], c='r', s=1)

cX, cY, cZ = twoDGaussianPlot(nData[0], nData[0], Means[0,:], CovMatrices[0,:,:])
plt.contour(cX, cY, cZ, 4)
cX, cY, cZ = twoDGaussianPlot(nData[1], nData[1], Means[1,:], CovMatrices[0,:,:])
plt.contour(cX, cY, cZ, 4)
cX, cY, cZ = twoDGaussianPlot(nData[2], nData[2], Means[2,:], CovMatrices[0,:,:])
plt.contour(cX, cY, cZ, 4)

plt.title('31240232-Gaussian Samples', fontsize=14)
plt.xlabel('x0')
plt.ylabel('x1')
plt.grid(True)
plt.gca().legend(('X0','X1','X2'))
#plt.savefig('1-gaussianSamples.png')
plt.savefig('7-gaussianSamples.png')
# Append into an array for the data we need
#
X = np.append(np.append(X0, X1, axis=0), X2, axis=0)


# In[]: prepare for my k-means algorithm


# Number of clusters
k = 3
# Set the random centroids
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(X), size=k)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(X), size=k)
# zip 함수를 이용해서 C_x, C_y 를 하나로 엮음.
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
# 만약에 C가 모두 0인상태로 시작하면 에러남 
# C가 하나라도 중복되면 에러
#C = np.zeros((3,2) 
print(C)

# Plotting along with the Centroids
#
#fig = plt.figure()
#plt.subplots(figsize=(7,5))
#plt.xlim([-6, 9])
#plt.ylim([-5, 8])
#
#plt.scatter(X0[:,0], X0[:,1], c='y', s=1)
#plt.scatter(X1[:,0], X1[:,1], c='y', s=1)
#plt.scatter(X2[:,0], X2[:,1], c='y', s=1)
#plt.scatter(C_x, C_y, marker='*', s=200, c='g')
#plt.title('31240232-Samples & Initial centres', fontsize=14)
#plt.xlabel('x0')
#plt.ylabel('x1')
#plt.grid(True)
#plt.gca().legend(('X0','X1','X2','Centres'))
#plt.savefig('2-samplesAndCetnres.png')


# In[]: myK-means algorithm


# To store the value of centroids when it updates
C_prev = np.zeros(C.shape) # 이전 값 저장.
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = EuclDist(C, C_prev, None)
# Loop will run till the error becomes zero
itcnt=0;
fig = plt.figure()
plt.subplots(figsize=(7,5))
plt.xlim([-6, 9])
plt.ylim([-5, 8])
#cX, cY, cZ = twoDGaussianPlot(nData[0], nData[0], Means[0,:], CovMatrices[0,:,:])
#plt.contour(cX, cY, cZ, 4)
#cX, cY, cZ = twoDGaussianPlot(nData[1], nData[1], Means[1,:], CovMatrices[1,:,:])
#plt.contour(cX, cY, cZ, 4)
#cX, cY, cZ = twoDGaussianPlot(nData[2], nData[2], Means[2,:], CovMatrices[2,:,:])
#plt.contour(cX, cY, cZ, 4)
plt.scatter(C[:, 0], C[:, 1], marker='*', s=150, c='r')
while error != 0:
    itcnt += 1
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = EuclDist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    # deepcopy를 해야지 C값을 변경해도 C_old값이 변하지 않음.
    C_prev = deepcopy(C)
    
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    # C_old는 C를 deepcopy한 값이다. C값과 C_old값이 같다는 것은
    # C값의 변동이 없다는 것, 
    plt.scatter(C[:, 0], C[:, 1], marker='*', s=50, c='m')
    error = EuclDist(C, C_prev, None)
# how many iterations in this algorithm

# clustering을 위한 색 index 지정
colors = ['y', 'c', 'k', 'g', 'b', 'm']
# plot the result of my algorithm

for i in range(k): # i, from 0 to 2 (because k = 3)
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        plt.scatter(points[:, 0], points[:, 1], s=1, c=colors[i])
        
plt.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='g')
#plt.title('31240232-Clustering&Centres', fontsize=14)
plt.title('31240232-Results-Covariance', fontsize=14)
plt.xlabel('x0')
plt.ylabel('x1')
plt.grid(True)

plt.savefig('7-compareCovariance.png')
#plt.savefig('4-2-compareCovariance.png')
#plt.savefig('3-clusteringAndCentres.png')

#fig = plt.figure()
#plt.subplots(figsize=(7,5))
#plt.xlim([-6, 9])
#plt.ylim([-5, 8])
#for i in range(k): # i, from 0 to 2 (because k = 3)
#        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
#        plt.scatter(points[:, 0], points[:, 1], s=1, c=colors[i])
#plt.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='g')
#plt.title('31240232-My K-means', fontsize=14)
#plt.xlabel('x0')
#plt.ylabel('x1')
#plt.grid(True)
#plt.gca().legend(('X0','X1','X2','Result centres'))

#plt.savefig('4-Myalgorithm.png')

# In[]: K-means algorithm inside sklearn (package)
#
#           
from sklearn.cluster import KMeans

# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
tt = kmeans.n_iter_

# Centroid values
centroids = kmeans.cluster_centers_
#
#
#
#fig = plt.figure()
#plt.subplots(figsize=(7,5))
#plt.xlim([-6, 9])
#plt.ylim([-5, 8])
#
#plt.scatter(X[:, 0], X[:, 1], c=labels, s=1, cmap='viridis')
#plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=200, marker='*');
#
#plt.title('31240232-sklearn K-means', fontsize=14)
#plt.xlabel('x0')
#plt.ylabel('x1')
#plt.grid(True)
#plt.gca().legend(('X','Result centres'))
#
##plt.savefig('5-sklearn.png')
#
## Comparing with scikit-learn centroids
#print('Original - Centres: ')
#print(Means)
#print('My algorithm - Centres: ')
#print(C) # From Scratch
#print('sklearn Kmeans - Centres: ')
#print(centroids) # From sci-kit learn
#
#print('How many iterations for K-means clustering')
#print('>> myAlgo: ', itcnt, '//', 'sklearn: ', tt)