import json
import random

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy
from sklearn_extra.cluster import KMedoids

df = pd.read_csv("fifaStats.csv")
#Gold players only (above 75 overall)
df = df[df['Overall'] > 74]
df = df.reset_index(drop=True)

print("\n FORWARDS: \n")
#seperate forwards from the entire dataset
forwards = df[df['Position'].isin(['ST', 'CF', 'LW', 'RW', 'LM', 'RM'])]
forwards = forwards[['Name', 'Overall', 'Acceleration', 'Crossing', 'Sprint', 'Positioning', 'Finishing', 'Shot', 'Long', 'Penalties','Agility', 'Balance', 'Reactions', 'Ball', 'Composure', 'Heading', 'Jumping','Stamina', 'Strength', 'Aggression', 'Skill moves', 'Passing', 'Dribbling']]
forwards = forwards[forwards['Overall'] > 75]
forwards = forwards.reset_index(drop = True)

fwd_stats = forwards[['Acceleration', 'Crossing', 'Sprint', 'Positioning', 'Finishing', 'Shot', 'Long', 'Penalties','Agility', 'Balance', 'Reactions', 'Ball', 'Composure', 'Heading', 'Jumping','Stamina', 'Strength', 'Aggression', 'Skill moves', 'Passing', 'Dribbling']]

init_list = []
for idx in range(len(forwards)):
    if((forwards.loc[idx, 'Name'] == "Kylian Mbappé") | (forwards.loc[idx, 'Name'] == "Erling Haaland") | (forwards.loc[idx, 'Name'] == "Ángel Di María")):
        init_list.append(fwd_stats.loc[idx])
        
       #print("HERE: ", forwards.loc[idx])
       #print("\n STATS: ", fwd_stats.loc[idx])
init_arr = np.array(init_list)
#print(init_arr)


model = KMedoids(n_clusters = 3, metric = "cosine", init = init_arr, random_state=None)
model.fit(fwd_stats)

cluster_labels = model.predict(fwd_stats) 

player_cluster_df = pd.DataFrame(cluster_labels, index=fwd_stats.index, columns=["cluster"])
player_cluster_df["cluster"].value_counts()

for cluster,players in player_cluster_df.groupby("cluster"):
    print("Cluster:", cluster, "Size:", players.shape[0])
    
    top_stats = fwd_stats.loc[players.index].sum().sort_values(ascending=False).head(15) / players.shape[0]
    print("\t", "Top Statistics:")
    for this_s, rate in top_stats.items():
        print("\t\t", this_s, "[%0.4f]" % rate)
    
    print("\t", "Player Sample:")
    for m_id in players.sample(15).index:
        if(forwards.loc[m_id, 'Overall'] > 78):
            print("\t\t", "Player", forwards.loc[m_id, 'Name'], " | Overall", forwards.loc[m_id, 'Overall'])

# Midfielders
midfielders = df[df['Position'].isin(['CAM', 'CM', 'CDM'])]
midfielders = midfielders[['Name', 'Overall', 'Acceleration', 'Sprint', 'Positioning', 'Finishing', 'Shot', 'Long', 'Volleys', 'Penalties', 'Vision', 'Crossing', 'Free', 'Curve', 'Agility', 'Balance', 'Reactions', 'Ball', 'Composure', 'Interceptions', 'Heading', 'Def', 'Standing', 'Sliding', 'Jumping', 'Stamina', 'Strength', 'Aggression', 'Skill moves', 'Passing', 'Dribbling']]
midfielders = midfielders.reset_index(drop = True)

mid_stats = midfielders[['Acceleration', 'Sprint', 'Positioning', 'Finishing', 'Shot', 'Long', 'Volleys', 'Penalties', 'Vision', 'Crossing', 'Free', 'Curve', 'Agility', 'Balance', 'Reactions', 'Ball', 'Composure', 'Interceptions', 'Heading', 'Def', 'Standing', 'Sliding', 'Jumping', 'Stamina', 'Strength', 'Aggression', 'Skill moves', 'Passing', 'Dribbling']]

init_mid_list = []
for idx in range(len(midfielders)):
    if((midfielders.loc[idx, 'Name'] == "Kevin De Bruyne") | (midfielders.loc[idx, 'Name'] == "Rodri") | (midfielders.loc[idx, 'Name'] == "Jude Bellingham")):
        init_mid_list.append(mid_stats.loc[idx])
        
       #print("HERE: ", forwards.loc[idx])
       #print("\n STATS: ", fwd_stats.loc[idx])
init_mid_arr = np.array(init_mid_list)

mid_model = KMedoids(n_clusters = 3, metric = "cosine", init = init_mid_arr, random_state=None)
mid_model.fit(mid_stats)

mid_labels = mid_model.predict(mid_stats) 

mid_cluster_df = pd.DataFrame(mid_labels, index=mid_stats.index, columns=["cluster"])
mid_cluster_df["cluster"].value_counts()
print("\n MIDFIELDERS: \n")
for cluster,players in mid_cluster_df.groupby("cluster"):
    print("Cluster:", cluster, "Size:", players.shape[0])
    
    top_stats = mid_stats.loc[players.index].sum().sort_values(ascending=False).head(10) / players.shape[0]
    print("\t", "Top Statistics:")
    for this_s, rate in top_stats.items():
        print("\t\t", this_s, "[%0.4f]" % rate)
    
    print("\t", "Player Sample:")
    for m_id in players.sample(15).index:
        if(midfielders.loc[m_id, 'Overall'] > 78):
            print("\t\t", "Player", midfielders.loc[m_id, 'Name'], " | Overall", midfielders.loc[m_id, 'Overall'])


# Defenders
defenders = df[df['Position'].isin(['CB', 'LB', 'RB', 'LWB', 'RWB'])]
defenders = defenders[['Name', 'Overall', 'Acceleration', 'Sprint', 'Positioning', 'Finishing', 'Shot', 'Long', 'Volleys', 'Penalties', 'Vision', 'Crossing', 'Free', 'Curve', 'Agility', 'Balance', 'Reactions', 'Ball', 'Composure', 'Interceptions', 'Heading', 'Def', 'Standing', 'Sliding', 'Jumping', 'Stamina', 'Strength', 'Aggression', 'Skill moves', 'Passing', 'Dribbling']]
defenders = defenders.reset_index(drop = True)

def_stats = defenders[['Acceleration', 'Sprint', 'Agility', 'Balance', 'Reactions', 'Ball', 'Composure', 'Interceptions', 'Heading', 'Def', 'Standing', 'Sliding', 'Jumping', 'Stamina', 'Strength', 'Aggression', 'Passing', 'Dribbling']]

init_def_list = []
for idx in range(len(defenders)):
    if((defenders.loc[idx, 'Name'] == "Virgil van Dijk") | (defenders.loc[idx, 'Name'] == "Theo Hernández")):
        init_def_list.append(def_stats.loc[idx])
        
       #print("HERE: ", forwards.loc[idx])
       #print("\n STATS: ", fwd_stats.loc[idx])
init_def_arr = np.array(init_def_list)

def_model = KMedoids(n_clusters = 2, metric = "cosine", init = init_def_arr, random_state=None)
def_model.fit(def_stats)

mid_labels = def_model.predict(def_stats) 

print("\n DEFENDERS: \n")

def_cluster_df = pd.DataFrame(mid_labels, index=def_stats.index, columns=["cluster"])
def_cluster_df["cluster"].value_counts()
for cluster,players in def_cluster_df.groupby("cluster"):
    print("Cluster:", cluster, "Size:", players.shape[0])
    
    top_stats = def_stats.loc[players.index].sum().sort_values(ascending=False).head(10) / players.shape[0]
    print("\t", "Top Statistics:")
    for this_s, rate in top_stats.items():
        print("\t\t", this_s, "[%0.4f]" % rate)
    
    print("\t", "Player Sample:")
    for m_id in players.sample(15).index:
        if(defenders.loc[m_id, 'Overall'] > 78):
            print("\t\t", "Player", defenders.loc[m_id, 'Name'], " | Overall", defenders.loc[m_id, 'Overall'])
