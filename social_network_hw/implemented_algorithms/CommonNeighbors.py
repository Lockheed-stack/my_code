#%%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# %%
def common_neighbors(G,node_x):
    CM_index={}
    for node_y in nx.nodes(G):
        if node_y!=node_x:
            cm_value=0
            for node in nx.neighbors(G,node_y):
                if node in nx.neighbors(G,node_x):
                    cm_value+=1
            CM_index[node_y]=cm_value
    
    trimed_CM_index = {}
    for node in CM_index.keys():
        if node not in nx.neighbors(G,node_x):
            if node!=0:
                trimed_CM_index[node]=CM_index[node]
    
    return trimed_CM_index
# %%
G = nx.read_adjlist('./SCHOLAT Link Prediction/train.csv',nodetype=int,delimiter=',')
common_neighbors(G,256)
# %%
