#%%
import networkx as nx
import pandas as pd 
import numpy as np
#%%
def AUC(output,G_test,node):
    '''
    @return auc score
    @param:
        @output: list-like, list[set(),set(),..,set()]
        @G_test: the graph for test
        @node: start node
    '''
    missing_edge={}
    for i in output:
        if i[0] in nx.neighbors(G_test,node):
            missing_edge[i[0]]=i[1]
    
    n1,n2,compare_count= 0,0,0
    for edge in missing_edge.items():
        for i in output:
            if i[0] not in nx.neighbors(G_test,node):#non-existent edge
                compare_count+=1
                if edge[1]<i[1]:
                    continue
                elif edge[1]>i[1]:
                    n1+=1
                else:
                    n2+=1
    n = compare_count
    if n==0:
        auc=0
    else:
        auc = (n1+0.5*n2)/n
    #print(f'node {node}\'s AUC: {auc}')
    return auc
#%%
def precision(top_L,output,G_test,node):
    '''
    @return: return a ratio 
    @top_L: choose top-L from output score. If top-L in (0,1) then choose the %top-L output score.
    @output: the score of link prediction
    '''
    if (top_L > 0) and (top_L < 1):
        top_L = int(len(output)*top_L)
        
    if top_L==0:
        return 0
    
    top_L_output = output[:top_L]
    top_L_dict = {}
    for i in top_L_output:
        top_L_dict[i[0]]=1
    L_r = 0
    for nbr in nx.neighbors(G_test,node):
        if nbr in top_L_dict:
            L_r+=1
    return L_r/top_L
# %%
def random_remove_edges(G,ratio):
    '''
    @return a nx.Graph which has been removed %ratio edges and a sample edge list
    '''
    edges = nx.edges(G)
    df = pd.DataFrame(np.array(edges))
    sample_edge = df.sample(frac=ratio,replace=False)
    ebunch =[]
    
    for edge in sample_edge.values:
        ebunch.append(tuple(edge))
    G_train = nx.Graph(G)
    G_train.remove_edges_from(ebunch)
    
    return G_train,ebunch