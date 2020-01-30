# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 01:56:51 2019

@author: taebe
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

df = pd.read_csv('iris2.csv')

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

NumDataPerClass = 50
X = df.iloc[0:100, [0,2]].values

#plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
#plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
#plt.xlabel('petal length')
#plt.ylabel('sepal length')
#plt.legend(loc='upper left')
#plt.show()

# --------------------------------------------------------------------
O = np.ones((2*NumDataPerClass, 1))
Y1 = np.zeros((50, 2))
Y2 = np.zeros((50, 2))
Y1[:,0] = X[:50, 0]
Y1[:,1] = X[:50, 1]

Y2[:,0] = X[50:100, 0]
Y2[:,1] = X[50:100, 1]

Y = np.concatenate((Y1, Y2), 0)
Y = np.append(Y, O, axis=1)
# --------------------------------------------------------------------
labelPos = np.ones(NumDataPerClass)
labelNeg = -1.0*np.ones(NumDataPerClass)
f = np.concatenate((labelPos, labelNeg))
# --------------------------------------------------------------------
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
# --------------------------------------------------------------------
a = np.random.randn(3) # weight 'a'

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
      
xx = np.arange(4, 10)
yy = -(a[0]/a[1])*xx - (a[2]/a[1])
#plt.plot(xx,yy, c='k')
plt.clf()
plt.scatter(Y_train[:,0], Y_train[:,1], s=3, c='r')
plt.scatter(Y_test[:,0], Y_test[:,1], s=3, c='b')
plt.plot(xx,yy, c='k')
plt.axis([4, 8, 0, 6])
plt.grid(True)
plt.title('31240232 / Sunwung Lee')
plt.gca().legend(('classifier', 'Training Set','Test Set'))
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.savefig('3_result')
plt.clf()

# ---------------------------------------------
plt.plot(range(MaxIter), P_train, 'b', range(MaxIter), P_test, 'r')
plt.grid(True)
plt.title('31240232 / Sunwung Lee')
plt.gca().legend(('Training Set', 'Test Set'))
plt.savefig('3_percentage_correct')
plt.clf()

plt.scatter(Y_train[:,0], Y_train[:,1], s=3, c='r')
plt.plot(xx,yy, c='k')

