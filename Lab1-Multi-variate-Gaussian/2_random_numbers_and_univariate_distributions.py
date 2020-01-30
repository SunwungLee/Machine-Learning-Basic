# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:40:50 2019

@author: taebe
"""
import numpy as np
import matplotlib.pyplot as plt
#import time
## <2. Random Numbers and Univariate Distributions>
#x = np.random.randn(10000,1) # just random number 0~1, float, not Gaussian
#y = np.random.rand(10001,1)
#plt.figure(figsize=(5,5)) # figsize: how big of figure size
##n, bins, patches = plt.hist(x, bins=20, color='m', alpha=0.8, rwidth=0.8)
#plt.hist([x, y], bins=100, label=['randn','rand'])
#
## RETURN
## n: array or list of arrays & the density and weight, 
## bins: nbins+1, always a single array, patches: list or list of lists
## INPUT
## bins: the number of bars(not affect the values), alpha: colour density 
## rwidth: gap among the bars
#print(n)
#print(bins)

## ---------------------------------------------
#MaxTrials = 10 # operate 10 times
#NumSamples = 1000 # total number of samples
#NumBins = 20 # the number of bins
#for trial in range(MaxTrials) :
##    plt.clf()
#    x = np.random.randn(NumSamples,1) # make 200(NumSamples) random numbers
#    counts, bins, patches = plt.hist(x, NumBins)
##    time.sleep(.5)
#    plt.clf()
#    print('Variation within bin counts: ', np.var(counts/NumSamples))
## np.var: calculate variation

## ---------------------------------------------
N = 1000;
x1 = np.zeros(N)
y1 = np.zeros(N)
for n in range(N) :
    x1[n] = np.sum(np.random.rand(12,1))-np.sum(np.random.rand(12,1));
    y1[n] = np.sum(np.random.rand(48,1))-np.sum(np.random.rand(48,1));

#plt.hist(x1, 40, color='b', alpha=0.8, rwidth=0.8)
plt.hist([x1,y1], 40, label=('#12','#48'))
plt.xlabel('Bin', FontSize=16)
plt.ylabel('Counts', FontSize=16)
plt.grid(True)
#
plt.legend()
plt.title('Histogram, 31240232/Sunwung Lee', fontsize=14)
#plt.savefig('2_rand_randn_2')
## ---------------------------------------------
