import json
import random

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


df = pd.read_csv("NBA_2024_STATS.csv")
pointGuards = df[df['Pos'] == 'PG']
starting_PGs = pointGuards[pointGuards['MP'] > 25]
features = starting_PGs
features = features.drop(['Player', 'Pos','G','GS', 'Tm', 'Age', 'MP','FG','FGA','FG%','3P','2P','ORB','DRB'], axis = 1)
features = features.dropna()
features = features.reset_index(drop =True)
starting_PGs = starting_PGs.reset_index(drop =True)
print(features.head())
#FG,FGA,FG%,3P,2P,ORB,DRB
from sklearn.cluster import KMeans
k = 4 
 
cluster_model = KMeans(n_clusters=k)
cluster_model.fit(features)
KMeans(n_clusters=k)
 
cluster_labels = cluster_model.predict(features)
player_cluster_df = pd.DataFrame(cluster_labels, index=features.index, columns=["cluster"])
player_cluster_df["cluster"].value_counts()

for cluster,players in player_cluster_df.groupby("cluster"):
    print("Cluster:", cluster, "Size:", players.shape[0])
    
    top_stats = features.loc[players.index].sum().sort_values(ascending=False).head(5) / players.shape[0]
    print("\t", "Top Statistics:")
    for this_s, rate in top_stats.items():
        print("\t\t", this_s, "[%0.4f]" % rate)
    
    print("\t", "Player Sample:")
    for m_id in players.sample(len(players)).index:
        print("\t\t", "Player", starting_PGs.loc[m_id, 'Player'])
'''
# w3schools elbow method implementation
intertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(features)
    intertias.append(kmeans.inertia_)
plt.plot(range(1,11), intertias, marker ='o')
plt.title('Elbow Method')
plt.xlabel('# Clusters')
plt.ylabel('Inertia')
plt.show()
# End w3schools elbow implementation
# We get k = 3
kmeans = KMeans(n_clusters=3)
# Features includes all column header besides Player, Pos, Tm, and Age
kmeans.fit(features)
labels = kmeans.predict(features)
#print(labels)
pg_cluster_df = pd.DataFrame(labels, index=starting_PGs.index, columns=["cluster"])
print(pg_cluster_df.head())
for cluster, players in pg_cluster_df.groupby("cluster"):
    print("Cluster: ", cluster, "Size: ", players.shape[0])
    for Name in players.sample().index:
        print("\t", Name)
'''