#%%
import algorithm 
import LTD1DT
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
# %%
G_scale_free  = nx.barabasi_albert_graph(500,1)
G_small_world = nx.watts_strogatz_graph(500,4,0.2,)
G_netscience = nx.read_adjlist('./trimed_netScience.csv',nodetype=int,delimiter=',')
G_uspower = nx.read_adjlist('./USpowerGrid.mtx',nodetype = int,)
# %%
model_sf = LTD1DT.model_V2(G_scale_free,False,[],[])
model_sw = LTD1DT.model_V2(G_small_world,False,[],[])
model_netsci = LTD1DT.model_V2(G_netscience,False,[],[])
model_us = LTD1DT.model_V2(G_uspower,False,[],[])
# %%
def calc_avg_R_diffu_num(model:LTD1DT.model_V2,run_times:int=100,seed_R_num:int=3,seed_T:list=[],rand_choose_R:bool=True):
   
    diffu_result_num = []
    
    if rand_choose_R:
        df_node = pd.DataFrame(model.G.nodes())
        with tqdm(total=run_times) as qbar:
            for i in range(run_times):
                # generate seed R
                rand_seed_R = df_node.sample(seed_R_num,replace=False).to_numpy(dtype=int)
                seed_R = np.reshape(rand_seed_R,rand_seed_R.shape[0])

                # test_model = model.copy()
                # test_model.update_seed_R(seed_R,True)
                # test_model.update_seed_T(seed_T,True)
                test_model = LTD1DT.model_V2(model.G,False,seed_T,seed_R)
                res = test_model.diffusion()

                diffu_result_num.append(len(res[2]))
                qbar.update(1)
    else:
        sorted_node_deg = sorted(list(nx.degree(model.G)),key=lambda x:x[1],reverse=True)
        seed_R = []
        for i in range(seed_R_num):
            seed_R.append(sorted_node_deg[i][0])

        # test_model = model.copy()
        # test_model.update_seed_R(seed_R,True)
        # test_model.update_seed_T(seed_T,True)
        
        # res = test_model.diffusion()
        
        # diffu_result_num.append(len(res[2]))
        with tqdm(total=run_times) as qbar:
            for i in range(run_times):
                # generate seed R

                test_model = LTD1DT.model_V2(model.G,False,seed_T,seed_R)
                #test_model.update_seed_R(seed_R,True)
                #test_model.update_seed_T(seed_T,True)
                
                res = test_model.diffusion()

                diffu_result_num.append(len(res[2]))
                qbar.update(1)

    return np.average(diffu_result_num)
#%%
calc_avg_R_diffu_num(model_netsci,100,3,[],False)
# %%
list(nx.degree(G_netscience))
# %%
max(nx.degree(G_scale_free),key=lambda x:x[1])
# %%
def choose_top_k(k:int,node_deg_list:list):
    '''choose top k nodes with highest degree

    parameters:
    -------------
    k: int
        choose k nodes
    node_deg_list: list
        A list of elements with tuples. tuple:(node,degee)
    
    return:
    -------------
    node_list: list
        top k nodes of list.
    '''
    left = 0
    right = len(node_deg_list)-1
    target = len(node_deg_list)-k

    while(True):
        pivot = partition(node_deg_list,left,right)

        if pivot == target:
            return 

def partition(node_deg_list:list,left:int,right:int):
    pass
#%%
model_sf.update_seed_R([1,0,27],True)
res = model_sf.diffusion()
res[2]
# %%
len(res[2])
# %%
sorted(list(nx.degree(G_scale_free)),key=lambda x:x[1],reverse=True)
# %%
