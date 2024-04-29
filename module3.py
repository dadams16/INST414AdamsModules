import json
import random

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy


df = pd.read_csv("fifaStats.csv")

#seperate forwards from the entire dataset
forwards = df[df['Position'].isin(['ST', 'CF', 'LW', 'RW', 'LM', 'RM'])]
forwards = forwards[['Name', 'Overall', 'Acceleration', 'Sprint', 'Positioning', 'Finishing', 'Shot', 'Long', 'Penalties','Agility', 'Balance', 'Reactions', 'Ball', 'Composure', 'Heading', 'Jumping','Stamina', 'Strength', 'Aggression', 'Skill moves', 'Passing', 'Dribbling']]
forwards = forwards.reset_index(drop = True)

fwd_stats = forwards[['Acceleration', 'Sprint', 'Positioning', 'Finishing', 'Shot', 'Long', 'Penalties','Agility', 'Balance', 'Reactions', 'Ball', 'Composure', 'Heading', 'Jumping','Stamina', 'Strength', 'Aggression', 'Skill moves', 'Passing', 'Dribbling']]

target_forward = 'Kylian Mbappé'
target_fwd = forwards[forwards['Name'] == target_forward]
target_stats = target_fwd[['Acceleration', 'Sprint', 'Positioning', 'Finishing', 'Shot', 'Long', 'Penalties','Agility', 'Balance', 'Reactions', 'Ball', 'Composure', 'Heading', 'Jumping','Stamina', 'Strength', 'Aggression', 'Skill moves', 'Passing', 'Dribbling']]

distances = scipy.spatial.distance.cdist(fwd_stats, target_stats, metric="cosine")[:,0]# I don't think this is the right syntax for cosine distance

query_distances = list(zip(forwards.index, distances))#may get indexing error
print("\nFORWARD SIMILARITY: \n")
for similar_player, similar_player_score in sorted(query_distances, key=lambda x: x[1], reverse=False)[:10]:
    cosine_similarity = abs(similar_player_score - 1)
    print(forwards.loc[similar_player, 'Name'], "| Overall: ", forwards.loc[similar_player, 'Overall'], "| Similarity: ", cosine_similarity)

# Midfielders
midfielders = df[df['Position'].isin(['CAM', 'CM', 'CDM'])]
midfielders = midfielders[['Name', 'Overall', 'Acceleration', 'Sprint', 'Positioning', 'Finishing', 'Shot', 'Long', 'Volleys', 'Penalties', 'Vision', 'Crossing', 'Free', 'Curve', 'Agility', 'Balance', 'Reactions', 'Ball', 'Composure', 'Interceptions', 'Heading', 'Def', 'Standing', 'Sliding', 'Jumping', 'Stamina', 'Strength', 'Aggression', 'Skill moves', 'Passing', 'Dribbling']]
midfielders = midfielders.reset_index(drop = True)

mid_stats = midfielders[['Acceleration', 'Sprint', 'Positioning', 'Finishing', 'Shot', 'Long', 'Volleys', 'Penalties', 'Vision', 'Crossing', 'Free', 'Curve', 'Agility', 'Balance', 'Reactions', 'Ball', 'Composure', 'Interceptions', 'Heading', 'Def', 'Standing', 'Sliding', 'Jumping', 'Stamina', 'Strength', 'Aggression', 'Skill moves', 'Passing', 'Dribbling']]

target_midfielder = 'Kevin De Bruyne'
target_mid = midfielders[midfielders['Name'] == target_midfielder]
target_mid_stats = target_mid[['Acceleration', 'Sprint', 'Positioning', 'Finishing', 'Shot', 'Long', 'Volleys', 'Penalties', 'Vision', 'Crossing', 'Free', 'Curve', 'Agility', 'Balance', 'Reactions', 'Ball', 'Composure', 'Interceptions', 'Heading', 'Def', 'Standing', 'Sliding', 'Jumping', 'Stamina', 'Strength', 'Aggression', 'Skill moves', 'Passing', 'Dribbling']]

mid_distances = scipy.spatial.distance.cdist(mid_stats, target_mid_stats, metric="cosine")[:,0]# I don't think this is the right syntax for cosine distance

query_distances = list(zip(midfielders.index, mid_distances))#may get indexing error
print("\nMIDFIELDER SIMILARITY: \n")

for similar_player, similar_player_score in sorted(query_distances, key=lambda x: x[1], reverse=False)[:10]:
    cosine_similarity = abs(similar_player_score - 1)
    print(midfielders.loc[similar_player, 'Name'], "| Overall: ", midfielders.loc[similar_player, 'Overall'], "| Similarity: ", cosine_similarity)

# Defenders
defenders = df[df['Position'].isin(['LWB', 'LB', 'CB', 'RB', 'RWB'])]
defenders = defenders[['Name', 'Overall', 'Acceleration', 'Sprint', 'Vision', 'Agility', 'Balance', 'Reactions', 'Ball', 'Composure', 'Interceptions', 'Heading', 'Def', 'Standing', 'Sliding', 'Jumping', 'Stamina', 'Strength', 'Aggression', 'Passing', 'Dribbling']]
defenders = defenders.reset_index(drop = True)

def_stats = defenders[['Acceleration', 'Sprint', 'Vision', 'Agility', 'Balance', 'Reactions', 'Ball', 'Composure', 'Interceptions', 'Heading', 'Def', 'Standing', 'Sliding', 'Jumping', 'Stamina', 'Strength', 'Aggression', 'Passing', 'Dribbling']]

target_defender = 'Virgil van Dijk'
target_def = defenders[defenders['Name'] == target_defender]
target_def_stats = target_def[['Acceleration', 'Sprint', 'Vision', 'Agility', 'Balance', 'Reactions', 'Ball', 'Composure', 'Interceptions', 'Heading', 'Def', 'Standing', 'Sliding', 'Jumping', 'Stamina', 'Strength', 'Aggression', 'Passing', 'Dribbling']]

def_distances = scipy.spatial.distance.cdist(def_stats, target_def_stats, metric="cosine")[:,0]# I don't think this is the right syntax for cosine distance

query_distances = list(zip(defenders.index, def_distances))#may get indexing error
print("\nDEFENDER SIMILARITY:\n")

for similar_player, similar_player_score in sorted(query_distances, key=lambda x: x[1], reverse=False)[:10]:
    cosine_similarity = abs(similar_player_score - 1)
    print(defenders.loc[similar_player, 'Name'], "| Overall: ", defenders.loc[similar_player, 'Overall'], "| Similarity: ", cosine_similarity)
