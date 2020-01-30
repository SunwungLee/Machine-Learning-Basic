# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 23:51:11 2019

@author: taebe
"""

import numpy as np
import matplotlib.pyplot as plt
# ---------------------------------------------
def PercentCorrect(Inputs, targets, weights): # How accurate is this
    # input, target, weight
    # Y_train, f_train, a
    N = len(targets) # the number of targets
    nCorrect = 0 # the number of correct data
    for n in range(N): # iteration
        OneInput = Inputs[n,:] # == OneInput = Inputs[n,] 
        if (targets[n] * np.dot(OneInput, weights) > 0):
            # what is the difference * and np.dot
            # targets[n] = constant, OneInput & weights = matrix
            nCorrect += 1
    return 100*nCorrect/N

# ---------------------------------------------
NumDataPerClass = 200

# Two-class problem, distinct means, equal covariance matrices
#
m1 = [[0, 5]]
m2 = [[5, 0]]
C = [[2, 1], [1, 2]]

# Set up the data
#
A = np.linalg.cholesky(C) # linear convert

X1 = np.random.randn(NumDataPerClass, 2) # operate random number (normal distribution)
Y1 = X1 @ A + m1 # linear convert
# [For report] How is the change of data Y1
#
#plt.scatter(X1[:,0],X1[:,1], s=3,c='r')
#plt.scatter(Y1[:,0],Y1[:,1], s=3,c='b')

X2 = np.random.randn(NumDataPerClass, 2)
Y2 = X2 @ A + m2
# ---------------------------------------------
plt.scatter(Y1[:,0], Y1[:,1], s=3, c='r')
plt.scatter(Y2[:,0], Y2[:,1], s=3, c='b')

Y = np.concatenate((Y1, Y2), 0) 
# index0~NumDataPerClass-1: Y1, NumDataP~2*Num: Y2
# concatenate((a1, a2, ...), axis=0, out=None)

# ---------------------------------------------
labelPos = np.ones(NumDataPerClass)
labelNeg = -1.0*np.ones(NumDataPerClass)
f = np.concatenate((labelPos, labelNeg))
# f matrix is consisted of 1 or -1

# ---------------------------------------------
# Generate random indices and order the data
#
rIndex = np.random.permutation(2*NumDataPerClass)
# permutation: list numbers in random rules
# if, np.random.permutation(5) --> [4 2 0 1 3]
#print('result of random.permutation: ', rIndex)

Yr = Y[rIndex, ] # the whole elements of Y, change the order of row vector
fr = f[rIndex] # == shuffle

# Training and test sets(half and half)
#
Y_train = Yr[0:NumDataPerClass]
f_train = fr[0:NumDataPerClass]

Y_test = Yr[NumDataPerClass:2*NumDataPerClass]
f_test = fr[NumDataPerClass:2*NumDataPerClass]
print(Y_train.shape, f_train.shape, Y_test.shape, f_test.shape)

Ntrain = NumDataPerClass
Ntest = NumDataPerClass
# ---------------------------------------------
# Random initialization of weights
#
a = np.random.randn(2) # weight 'a'
#print('a= ', a)
#xx = np.arange(-5, 10)
#yy = -(a[1]/a[0])*xx
#plt.plot(xx,yy,c='k')

# What is the performance with the initial random weights?
#
print('Initial Percentage Correct: ', PercentCorrect(Y_train, f_train, a))
# PercentCorrect(Inputs, targets, weights)

# Number of iterations and Learning rate
#
MaxIter = 400
alpha = 0.01

# Space for plots
#
P_train = np.zeros(MaxIter)
P_test = np.zeros(MaxIter)

# Main loop
#
for iter in range(MaxIter):
    # Select a data item at random
    #
    r = np.floor(np.random.rand()*Ntrain).astype(int) # convert float -> int
    y = Y_train[r,:]
    # Ntrain=200, 0<rand()<1, r is an index -> convert float to int
    # ex, np.floor(26.565) = 26.0 // ceil(), round(), floor()
    
    # If it is misclassified, update weights
    #
    if (f_train[r] * np.dot(y, a) < 0):
        a += alpha * f_train[r] * y
        
    # Evaluate training and test performances for plotting
    #
    P_train[iter] = PercentCorrect(Y_train, f_train, a);
    P_test[iter] = PercentCorrect(Y_test, f_test, a);
    
print('Percentage Correct After Training: ', 
      PercentCorrect(Y_train, f_train, a),'->', 
      PercentCorrect(Y_test, f_test, a));
# ---------------------------------------------
#
#xx = np.arange(-5, 10)
#yy = -(a[0]/a[1])*xx
#plt.plot(yy,xx, c='k')

#plt.scatter(Y_train[:,0], Y_train[:,1], s=3, c='r')
#plt.scatter(Y_test[:,0], Y_test[:,1], s=3, c='b')
#plt.axis([-5, 9, -5, 9])
#plt.grid(True)
#plt.title('31240232 / Sunwung Lee')
#plt.gca().legend(('Training Set', 'Test Set'))
##plt.savefig('partioning the data')
#plt.clf()
#
#plt.scatter(Y_train[:,0], Y_train[:,1], s=3, c='r')
#plt.plot(yy,xx, c='k')
#plt.axis([-5, 9, -5, 9])
#plt.grid(True)
#plt.title('31240232 / Sunwung Lee')
#plt.gca().legend(('classifier', 'Training Set'))
##plt.savefig('result of classifier-training')
#plt.clf()
#
#plt.scatter(Y_test[:,0], Y_test[:,1], s=3, c='b')
#plt.plot(yy,xx, c='k')
#plt.axis([-5, 9, -5, 9])
#plt.grid(True)
#plt.title('31240232 / Sunwung Lee')
#plt.gca().legend(('classifier', 'Test Set'))
##plt.savefig('result of classifier-test')
#plt.clf()
#
# ---------------------------------------------
#plt.plot(range(MaxIter), P_train, 'b', range(MaxIter), P_test, 'r')
#plt.grid(True)
#plt.title('31240232 / Sunwung Lee')
#plt.gca().legend(('Training Set', 'Test Set'))
#plt.savefig('percentage correct')

# ---------------------------------------------
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
model = Perceptron()
model.fit(Y_train, f_train) # SGD, 
# parameter: training data, target values
# return: self(returns an instance of self).
fh_train = model.predict(Y_test) 
# parameter: samples.
# return: predicted class label per sample
print(accuracy_score(f_test, fh_train))
# parameter: Ground truth labels, Predicted labels
# return: score (if normalize = True, return fraction of correctly classified samples(float), 
#         else returns the number of correctly classified samples(int))


#
#
#이 짓거리가 뭐냐 하면,
#model이란 이름의 Perceptron을 하나 만들어, 그리고 Y_train이랑 f_train을 사용해서 model을 만들어 --> training 시킴
#model.predict 함수 -> Y_train 샘플들 넣어놓고 예상되는 class 라벨들 따내 = fh_train이야
#그리고 이 fh_train이랑 f_train이랑 얼마나 똑같은지 비교해 
#


