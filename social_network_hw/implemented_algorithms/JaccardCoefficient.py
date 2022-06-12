import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# %%
#G = nx.read_adjlist('./SCHOLAT Link Prediction/train.csv',nodetype=int,delimiter=',')
# %%
def Jaccard(G,node_x):
    '''
    @return sorted_trimed_J_index, type:list[tuple]
    '''
    J_index={}
    for node_y in G.nodes():
        if node_y!=node_x:
            common_num=0
            all_nbrs_num = nx.degree(G,node_x)+nx.degree(G,node_y)
            for node in nx.neighbors(G,node_y):
                if node in nx.neighbors(G,node_x):
                    common_num+=1
            if all_nbrs_num==0:
                J_index[node_y]=0
            else:        
                J_index[node_y]=common_num/all_nbrs_num
            
    trimed_J_index={}
    for node in J_index.keys():
        if node not in nx.neighbors(G,node_x):
            #if J_index[node]!=0:
            trimed_J_index[node]=J_index[node]
    
    sorted_trimed_J_index=sorted(trimed_J_index.items(),key=lambda kv:kv[1],reverse=True)
    return sorted_trimed_J_index