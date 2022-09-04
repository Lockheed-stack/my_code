#%%
import networkx as nx
import numpy as np
import math
import pandas as pd
from tqdm import tqdm
# %%
def all_combination(num_of_selection:int,boundary:int):
    result:list = []
    chosen_index = np.arange(num_of_selection)
    end_index = (chosen_index+boundary-num_of_selection).copy()
    phase , loc = -2,-1

    while chosen_index[0]!=boundary-num_of_selection:

        while chosen_index[-1]!= boundary:
            result.append(chosen_index.copy())
            if (chosen_index[-1]+1)!= boundary:
                chosen_index[-1]+=1
            else:
                break

        if num_of_selection==1:#only select one
            return result
        
        else:#at least two selected
            
            # move to right location
            while chosen_index[loc]==end_index[loc] and phase<loc-1:
                loc-=1

            #big change
            if (loc-1) == phase and chosen_index[loc]==end_index[loc]:
                
                if chosen_index[phase]<end_index[phase]:
                    chosen_index[phase]+=1
                else:
                    #prevent out of bounds
                    if phase-1>= -(num_of_selection):
                        phase-=1
                    chosen_index[phase]+=1

                for i in np.arange(-1,phase,-1):
                    chosen_index[i]=chosen_index[phase]+(i-phase)
                
                loc=-1

            #small change
            else:
                chosen_index[loc]+=1
                for i in np.arange(-1,loc,-1):
                    chosen_index[i]=chosen_index[loc]+(i-loc)
                loc=-1

        if chosen_index[0]==end_index[0]:
            result.append(chosen_index.copy())

    return result
#%%
def isomorphism_check(subgraph_list,G_temp):

    if len(subgraph_list)==0:
        return False

    else:
        for G in subgraph_list:
            # Returns True if the graphs G1 and G2 are isomorphic and False otherwise
            if nx.is_isomorphic(G,G_temp):
                return True
    return False
#%%
def create_spanning_subgraph(G):
    spanning_subgraph_dict={}
    num_of_selections = int(nx.number_of_edges(G)/2)
    edge_list = list(nx.edges(G))
    selection_dict ={}

    #pick out all kinds of selections
    for i in range(1,num_of_selections+1):
        selection_dict[i]=all_combination(i,nx.number_of_edges(G))

    
    for select in selection_dict.items():
        with tqdm(total=len(select[1])) as qbar:
            subgraph1 = []
            subgraph2 = []
            for index in select[1]:
                selected_edge_1 = []
                selected_edge_2 = []
                # 得到一种组合便可得到两种选法
                
                # 第一种选法
                for i in index:
                    selected_edge_1.append(edge_list[i])
                G_temp = G.copy()
                G_temp.remove_edges_from(selected_edge_1)
                if isomorphism_check(subgraph1,G_temp) == False and nx.is_connected(G_temp):
                    subgraph1.append(G_temp)
                
                # 第二种选法
                for i in range(nx.number_of_edges(G)):
                    if i not in index:
                        selected_edge_2.append(edge_list[i])
                G_temp = G.copy()
                G_temp.remove_edges_from(selected_edge_2)
                if isomorphism_check(subgraph2,G_temp) == False and nx.is_connected(G_temp):
                    subgraph2.append(G_temp)   
                
                qbar.update(1)
                
            spanning_subgraph_dict[select[0]] = subgraph1
            spanning_subgraph_dict[nx.number_of_edges(G)-select[0]] = subgraph2
            
    
    return spanning_subgraph_dict
# %%
def random_create_spanning_subgraph(G:nx.Graph,target_num:int):
    
    if target_num>math.comb(nx.number_of_edges(G),int(nx.number_of_edges(G)/2)):
        print("invalid target num")
        return 

    spanning_subgraph_dict={}
    edge_list = list(nx.edges(G))
    edge_num = nx.number_of_edges(G)
    df = pd.DataFrame(edge_list)
    actually_get = 0
    curr_num = 0

    while(curr_num<target_num):
        select_num = np.random.randint(1,edge_num)
        select_np = df.sample(select_num,replace=False).to_numpy()
        selection_edges=[]
        if select_num not in spanning_subgraph_dict.keys():
            spanning_subgraph_dict[select_num]=[]

        for edge in select_np:
            selection_edges.append(tuple(edge))
        
        G_temp = G.copy()
        G_temp.remove_edges_from(selection_edges)
        if isomorphism_check(spanning_subgraph_dict[select_num],G_temp) == False and nx.is_connected(G_temp):
            spanning_subgraph_dict[select_num].append(G_temp)
            actually_get+=1
        curr_num+=1

    print(f"target:{target_num}, actually get:{actually_get}")
    return spanning_subgraph_dict
# %%
