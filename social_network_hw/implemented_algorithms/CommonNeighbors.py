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
G = nx.read_adjlist('../SCHOLAT Link Prediction/train.csv',nodetype=int,delimiter=',')
output = common_neighbors(G,489)
output = sorted(output.items(),key=lambda kv:kv[1],reverse=True)
# %%
def AUC(output):
    G_test = nx.read_adjlist('../SCHOLAT Link Prediction/test.csv',nodetype=int,delimiter=',')
    node = 4923
    miss_edge={}
    for i in output:
        if i[0] in nx.neighbors(G_test,node):
            miss_edge[i[0]]=i[1]
    n1,n2=0,0
    for edge in miss_edge.items():
        for i in output:
            if i[0] not in nx.neighbors(G_test,node):# non-existent edge
                if edge[1] <i[1]:
                    continue
                elif edge[1]>i[1]:
                    n1+=1
                else:
                    n2+=1
    
    n = len(miss_edge)*len(output)
    auc = (n1+0.5*n2)/n
    print(auc)
#%%
output
# %%
