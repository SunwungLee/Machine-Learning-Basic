import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
wine_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', \
              'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315',\
              'Proline']
wine_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', names = wine_names) 
wine_df = pd.DataFrame(wine_data)
wine_df.Class = wine_df.Class - 1

# 원본 class를 바탕으로 plotting
fig = plt.figure()
wine_df.plot.scatter(x = 'Alcohol', y = 'OD280/OD315', c= 'Class', figsize=(10,6), colormap='jet')
plt.title('31240232-Original Wine Data', fontsize=14)
plt.grid(True)
#plt.savefig('8-1-uci_winedata_ori.png')

# In[]: K-means algorithm

#kmeans = KMeans(n_clusters=3, init = 'random', max_iter = 1, random_state = 5).fit(wine_df.iloc[:,[12,1]])
#centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns = list(wine_df.iloc[:,[12,1]].columns.values))
#fig, ax = plt.subplots(1, 1)
#wine_df.plot.scatter(x = 'Alcohol', y = 'OD280/OD315', c= kmeans.labels_, figsize=(12,8), colormap='jet', ax=ax, mark_right=False)
#centroids_df.plot.scatter(x = 'Alcohol', y = 'OD280/OD315', ax = ax,  s = 80, mark_right=False)

kmeans = KMeans(n_clusters=3, init = 'random', max_iter = 100, random_state = 5).fit(wine_df.iloc[:,[12,1]])
centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns = list(wine_df.iloc[:,[12,1]].columns.values))
fig, ax = plt.subplots(1, 1, figsize=(10,6))
wine_df.plot.scatter(x = 'Alcohol', y = 'OD280/OD315', c= kmeans.labels_, figsize=(10,6), colormap='jet', ax=ax, mark_right=False)
centroids_df.plot.scatter(x = 'Alcohol', y = 'OD280/OD315', ax = ax,  s = 80, mark_right=False)
ax.set_title('31240232-clustered Data by using sklearn Kmeans algorithm', fontsize=14)
ax.grid(True)
#fig.savefig('8-2-uci_winedata_kmeans.png')

# In[]: Evaluating K-means algorithm

# Run the Kmeans algorithm and get the index of data points clusters
sse = []
list_k = list(range(1, 10))
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,10))

figcnt = 0;
figrow = 0;
figcol = 0;
for k in list_k:
    km = KMeans(n_clusters=k, init = 'random', max_iter = 100, random_state = 5).fit(wine_df.iloc[:,[12,1]])
    
    # draw each case by using subplot
    if k < 6 and k > 1:
        centroids_df = pd.DataFrame(km.cluster_centers_, columns = list(wine_df.iloc[:,[12,1]].columns.values))
        wine_df.plot.scatter(x = 'Alcohol', y = 'OD280/OD315', c= km.labels_, colormap='jet', ax=ax[figrow][figcol], mark_right=False)
        centroids_df.plot.scatter(x = 'Alcohol', y = 'OD280/OD315', ax=ax[figrow][figcol],  s = 80, mark_right=False)
        figcol += 1
        if figcol == 2:
            figrow = 1;
            figcol = 0;
    
    
    sse.append(km.inertia_)
#fig.savefig('8-4-CompareNumberofK.png')
# Plot sse against k
plt.figure(figsize=(7, 5))
plt.plot(list_k, sse, '-o')
plt.title('31240232-Sum of squared distance', fontsize=14)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')
plt.grid(True)
#plt.savefig('8-3-compareCovariance.png')

# In[]: compare with the original class



# labels 랑 wind_df.Class랑 비교하기
# cluster 개수 어떻게 정할 것인가 그래프 띄우면 좋을듯.