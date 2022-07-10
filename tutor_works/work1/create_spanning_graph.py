#%%
import networkx as nx
import numpy as np
# %%
G = nx.complete_graph(range(1,5))
nx.draw_networkx(G)
# %%
G1=nx.Graph(G)
G2=nx.Graph(G)
G1.remove_edge(1,2)
G2.remove_edge(3,4)
nx.draw_networkx(G1)
# %%
nx.draw_networkx(G2)
# %%
def all_combination(num_of_selection,boundary):
    result = []
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
def create_spanning_subgraph(G):
    spanning_subgraph=[]
    
    return spanning_subgraph
# %%
# %%
