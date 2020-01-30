# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:41:08 2019

@author: taebe
"""
import numpy as np
import matplotlib.pyplot as plt

# <3. Uncertainty in Estimation>
sampleSizeRange = np.linspace(100,200,40) 
# Return evenly spaced numbers over a specified interval.

plotVar = np.zeros(len(sampleSizeRange)) # make an array, values are all 0
for sSize in range(len(sampleSizeRange)):
    numSamples = np.int(sampleSizeRange[sSize])
    # make sampleSizeRange[sSize] integer (float -> integer)
    MaxTrial=2000
    vStrial=np.zeros(MaxTrial) # make an array
    for trial in range(MaxTrial): # operate 2000 times
        xx = np.random.randn(numSamples,1) # xx =  “standard normal”
        vStrial[trial] = np.var(xx) # 갯수에 따라 달라지는 var를 확인
    plotVar[sSize] = np.var(vStrial) # 결과로는 var이 낮아짐. 데이터가 밀집된다.
    
sampleSizeRange1 = np.linspace(100,300,40) 
# Return evenly spaced numbers over a specified interval.

plotVar1 = np.zeros(len(sampleSizeRange1)) # make an array, values are all 0
for sSize in range(len(sampleSizeRange1)):
    numSamples = np.int(sampleSizeRange1[sSize])
    # make sampleSizeRange[sSize] integer (float -> integer)
    MaxTrial=2000
    vStrial=np.zeros(MaxTrial) # make an array
    for trial in range(MaxTrial): # operate 2000 times
        xx = np.random.randn(numSamples,1) # xx =  “standard normal”
        vStrial[trial] = np.var(xx) # 갯수에 따라 달라지는 var를 확인
    plotVar1[sSize] = np.var(vStrial)

plt.plot(sampleSizeRange, plotVar, label='#200')
plt.plot(sampleSizeRange1, plotVar1, label='#300')


plt.grid(True)
#
plt.legend()
# ---------------------------------------------
