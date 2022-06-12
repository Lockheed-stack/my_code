import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# %%
#G = nx.read_adjlist('./SCHOLAT Link Prediction/train.csv',nodetype=int,delimiter=',')
# %%
def Adamic_Adar(G,node_x):
    ## Return a dict of Adamin Adar index
    AA_index={}
    for node_y in G.nodes():
        if node_y!=node_x:
            aa_value=0
            for node in nx.neighbors(G,node_y):
                if node in nx.neighbors(G,node_x):
                    aa_value+=1/np.log(nx.degree(G,node))
            AA_index[node_y]=aa_value
            
    trimed_AA_index={}
    for node in AA_index.keys():
        if node not in nx.neighbors(G,node_x):
            #if AA_index[node]!=0:
            trimed_AA_index[node]=AA_index[node]
    
    sorted_trimed_AA_index= sorted(trimed_AA_index.items(),key=lambda kv:kv[1],reverse=True)   
    return sorted_trimed_AA_index