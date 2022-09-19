#%%
import algorithm 
import LTD1DT
import networkx as nx
import numpy as np
import pandas as pd
# %%
scale_free_G  = nx.barabasi_albert_graph(500,1)
# %%
small_world = nx.watts_strogatz_graph(500,4,0.2,)
# %%
G_netscience = nx.read_adjlist('./trimed_netScience.csv',nodetype=int,delimiter=',')
G_uspower = nx.read_adjlist('./USpowerGrid.mtx',nodetype = int,)
# %%
model_sc = LTD1DT.model_V2(scale_free_G,False,[],[])
model_sw = LTD1DT.model_V2(small_world,False,[],[])
model_netsci = LTD1DT.model_V2(G_netscience,False,[],[])
model_us = LTD1DT.model_V2(G_uspower,False,[],[])
#%%
df_node = pd.DataFrame(scale_free_G.nodes(),)
df_node
# %%
rand_seed_R = df_node.sample(3,replace=False,ignore_index=True).to_numpy(dtype=int)
seed_R  = np.reshape(rand_seed_R,rand_seed_R.shape[0])
# %%
un_al = algorithm.unconstrained_algorithm()
un_al.MinGreedy(model_sc,3,seed_R)
# %%
un_al.ContrId(model_sc,3,seed_R)
# %%
