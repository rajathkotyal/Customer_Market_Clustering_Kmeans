# -*- coding: utf-8 -*-
'''
|**********************************************************************|
* Project           : Customer Segmentation using K-Means Clustering
*
* Program name      : customer_clustering.py
*
* Author            : Rajath_Kotyal
*
* Date created      : 15/07/2020
*
* Purpose           : A K-Means Clustering algorithm is built around a shopping mall dataset such that the
*                     customers are Seperated/Clustered into different catagories.
|**********************************************************************|

> Customer Data is taken from a Shopping Mall. which contains :
1.   Customer Age
2.   Customer Annual Salary
3.   Spending score (1-100, 100 meaning the person is an avid shopper)
[ This dataset was used as it is simple & easy to understand for the viewers, please feel free to use datasets with more number of features ]'''

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Importing the dataset
url = 'https://raw.githubusercontent.com/rajathkotyal/Customer_Market_Clustering_Kmeans/master/Mall_Customers.csv'
dataset = pd.read_csv(url)
X = dataset.iloc[: , 2:].values
y = dataset.iloc[:, 0]

#Analysing Corellation between the features

correl = dataset.corr()
f, ax = plt.subplots(figsize = (10,5))
sns.set(font_scale=1.2)
sns.heatmap(correl ,annot = True, annot_kws={'size':10})

#[Click here](https://towardsdatascience.com/formatting-tips-for-correlation-heatmaps-in-seaborn-4478ef15d87f) to learn about correlation heatmaps.
#Using the elbow method to find the optimal number of clusters

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11) :
  kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 69)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("Elbow Method")
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# WCSS = Within Cluster Sum of Squares

#Since the graph does not have a drastic downfall from around **point 5** , we will choose 5 clusters.
'''To read more about the Elbow method : [Click here](https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/)'''

## Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 69)
y_kmeans = kmeans.fit_predict(X)

#Creating & Exporting dataset with Clusters assigned
df = []
df = (np.c_[y, y_kmeans, X])
os.getcwd() # current working directory

#Exporting Dataset to csv
#The created file should be present in the sidebar to the left (Refresh if not present)
np.savetxt("clusterDataset.csv", df, delimiter=",", fmt='%s')

# Visualising the clusters
## Plotting graph between only 2 features as its hard for us to visualise data in multi dimensions.

plt.scatter(X[ y_kmeans == 0 , 1 ] , X[y_kmeans == 0 , 2] , s = 100, c = 'red' , label = 'cluster 1')
plt.scatter(X[ y_kmeans == 1 , 1 ] , X[y_kmeans == 1 , 2] , s = 100, c = 'blue' , label = 'cluster 2')
plt.scatter(X[ y_kmeans == 2 , 1 ] , X[y_kmeans == 2 , 2] , s = 100, c = 'green' , label = 'cluster 3')
plt.scatter(X[ y_kmeans == 3 , 1 ] , X[y_kmeans == 3 , 2] , s = 100, c = 'cyan' , label = 'cluster 4')
plt.scatter(X[ y_kmeans == 4 , 1 ] , X[y_kmeans == 4 , 2] , s = 100, c = 'magenta' , label = 'cluster 5')

plt.scatter(kmeans.cluster_centers_[ : , 1], kmeans.cluster_centers_[ : , 2], s=300, c = 'yellow', label = 'centroid')
plt.title("Clusters")
plt.xlabel("Annual Income")
plt.ylabel("Spending score")
plt.legend()
plt.show()


'''Please open the ipynb file in Google Colaboratory for a better UI.'''
