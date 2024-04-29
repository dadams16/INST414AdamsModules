import json
import random

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv('International_Report_Departures.csv')

df = df.drop(['data_dte', 'Month', 'airlineid', 'carrier', 'carriergroup', 'type', 'Scheduled', 'Charter'], axis = 1)
df = df.sort_values(by = ['Year'], ascending = False)
df = df[df['Year'] > 2018]

df = df.reset_index(drop = True)

print("NUMBER: ", range(len(df)))

print(df.head())

g = nx.Graph() # Build the graph

for idx in range(len(df)):
    # Get the airport id and name of departure location
    departure_id = df.loc[idx, "usg_apt_id"]
    departure_name = df.loc[idx, "usg_apt"]
    #print("Name #", idx, ": ", departure_name)
        
    # Get the airport id and name of destination location
    destination_id = df.loc[idx, 'fg_apt_id']
    destination_name = df.loc[idx, 'fg_apt']

    # For each departure location, make it a node
    g.add_node(departure_id, name = departure_name)
    g.add_node(destination_id, name = destination_name)

    # Each line has one departing location and one destination location.
    # Make this an edge
    current_weight = g.get_edge_data(departure_id, destination_id, default={"weight":0})["weight"]
    this_edge = g.get_edge_data(departure_id, destination_id)
    if this_edge is None:
        numFlights = df.loc[idx, 'Total']
        g.add_edge(departure_id, destination_id, weight=numFlights)
    else:
        g[departure_id][destination_id]["weight"] += numFlights


# check number of nodes
print("Nodes: ", len(g.nodes))
# check number of edges
print("Edges: ", len(g.edges))
# create .graphml file
nx.write_graphml(g, "airports.graphml")

top_k = 20 # how many of the most central nodes to print

centrality_degree = nx.degree_centrality(g)

bw_centrality = nx.betweenness_centrality(g)



for u in sorted(centrality_degree, key=centrality_degree.get, reverse=True)[:top_k]:
    print(u, g.nodes[u]['name'], centrality_degree[u])
    print("")
print("------------"*5)
for u in sorted(bw_centrality, key=bw_centrality.get, reverse=True)[:top_k]:
    print(u, g.nodes[u]['name'], bw_centrality[u])
    print("")
print("------------"*5)

