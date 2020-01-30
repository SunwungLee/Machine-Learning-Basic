# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 23:55:57 2019

@author: taebe
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from mpl_toolkits.mplot3d import Axes3D
import sklearn.metrics as sm

#Load iris dataset
from sklearn import datasets
iris = datasets.load_iris()

#Scale the variables and load into 'X'
from sklearn.preprocessing import scale
X = scale(iris.data)

#Assign target values to 'y'
y = pd.DataFrame(iris.target)

#Create an object for our variable names
variable_names = iris.feature_names

#Print first 10 records
print(X[0:10,])

#Import KMeans model and instantiate it.
from sklearn.cluster import KMeans
clustering = KMeans(n_clusters=3, random_state=5)

#Use fit method to create a model
clustering.fit(X)

# In[]:

#Create dataframe for original data comparison with model prediction
iris_df = pd.DataFrame(iris.data)
iris_df.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
y.columns = ['Targets']

#Set a variable color_theme to color our data points by their species label. 
color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue'])

#Set the color_theme as per original dataset labels 'iris.target'.
plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width, c=color_theme[iris.target], s=50)
plt.title('Original Dataset Labels')

#Change the color_theme as per the labels that were predicted by our clustering model.
plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width, c=color_theme[clustering.labels_], s=50)
plt.title('K-Means Classification')
plt.show()

# In[]:
#Create dataframe for original data comparison with model prediction
iris_df = pd.DataFrame(iris.data)
iris_df.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
y.columns = ['Targets']

#Set a variable color_theme to color our data points by their species label. 
color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue'])

relabel = np.choose(clustering.labels_, [2, 0, 1]).astype(np.int64)
plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width, c=color_theme[iris.target], s=50)
plt.title('Original Dataset Labels')

#Change the color_theme as per the new object 'relabel'.
plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width, c=color_theme[relabel], s=50)
plt.title('K-Means Classification')
plt.show()

# In[]:
from sklearn.metrics import classification_report
print(classification_report(y, relabel))