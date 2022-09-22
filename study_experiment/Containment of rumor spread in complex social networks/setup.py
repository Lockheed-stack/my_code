#%%
from pdb import line_prefix
from algorithm import unconstrained_algorithm as un_al
from LTD1DT import model_V2
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore',category=FutureWarning,)
# %%
G_scale_free  = nx.barabasi_albert_graph(500,1)
G_small_world = nx.watts_strogatz_graph(500,4,0.2,)
G_netscience = nx.read_adjlist('./trimed_netScience.csv',nodetype=int,delimiter=',')
G_uspower = nx.read_adjlist('./USpowerGrid.mtx',nodetype = int,)
# %%
model_sf = model_V2(G_scale_free,False,[],[])
model_sw = model_V2(G_small_world,False,[],[])
model_netsci = model_V2(G_netscience,False,[],[])
model_us = model_V2(G_uspower,False,[],[])
# %%
def calc_avg_R_diffu_num(G:nx.Graph,run_times:int=100,seed_R_num:int=3,rand_choose_R:bool=True):
    ''' Using three different algorithms respectively to calculate the number of rumor diffusion.

    Parameters
    ----------
    G : nx.Graph
        A Networkx graph
    run_times : int, optional
        The number of simulation of diffusion times, by default 100
    seed_R_num : int, optional
        The rumor nodes number, by default 3
    rand_choose_R : bool, optional
        Decide whether to use random seed_R , by default True

    Returns
    -------
    (greedy_diffu_res,pagerank_diffu_res,contrid_diffu_res) : tuple
        The final diffusion result of choosing k(from 0 to 10) nodes of Truth of three algorithms respectively.
    '''
    # diffu_result_num = []
    algor = un_al()
    greedy_diffu_res = []
    pagerank_diffu_res = []
    contrid_diffu_res = []

    # select by random
    if rand_choose_R:
        df_node = pd.DataFrame(G.nodes())
        
        for k in range(0,11):# select k(from 0 to 10) nodes of Truth
            print(f'choose {k} Truth nodes...')
            greedy_num,pageranK_num,contrid_num=0,0,0

            with tqdm(total=run_times) as qbar:
                for i in range(run_times): # run 100 times
                    # generate seed R
                    rand_seed_R = df_node.sample(seed_R_num,replace=False).to_numpy(dtype=int)
                    seed_R = np.reshape(rand_seed_R,rand_seed_R.shape[0])
                    
                    # May be deprecated later
                    # test_model = model.copy()
                    # test_model.update_seed_R(seed_R,True)
                    # test_model.update_seed_T(seed_T,True)

                    test_model = model_V2(G,False,[],seed_R)
                    
                    # generate seed T
                    greedy_T_nodes = algor.MinGreedy(test_model,k,seed_R)
                    pagerank_T_nodes = algor.pagerank(G,k)
                    contrid_T_nodes = algor.ContrId(test_model,k,seed_R)
                    
                    # run diffusion function respectively

                    # greedy
                    test_model.update_seed_T(greedy_T_nodes,True)
                    greedy_num += len(test_model.diffusion()[2])
                    # pagerank
                    test_model.update_seed_T(pagerank_T_nodes,True)
                    pageranK_num += len(test_model.diffusion()[2])
                    # ContrId
                    test_model.update_seed_T(contrid_T_nodes,True)
                    contrid_num += len(test_model.diffusion()[2])

                    qbar.update(1)

            greedy_diffu_res.append(greedy_num/100)
            pagerank_diffu_res.append(pageranK_num/100)
            contrid_diffu_res.append(contrid_num/100)

        return greedy_diffu_res,pagerank_diffu_res,contrid_diffu_res
    
    
    # select by degree
    else:
        sorted_node_deg = sorted(list(nx.degree(G)),key=lambda x:x[1],reverse=True)
        seed_R = []
        for i in range(seed_R_num):
            seed_R.append(sorted_node_deg[i][0])

        # test_model = model.copy()
        # test_model.update_seed_R(seed_R,True)
        # test_model.update_seed_T(seed_T,True)
        
        # res = test_model.diffusion()
        
        # diffu_result_num.append(len(res[2]))

        for k in range(0,11):# select k(from 0 to 10) nodes of Truth
            print(f'choose {k} Truth nodes...')
            greedy_num,pageranK_num,contrid_num=0,0,0

            with tqdm(total=run_times) as qbar:
                for i in range(run_times):
                    # generate seed R

                    test_model = model_V2(G,False,[],seed_R)
                    #test_model.update_seed_R(seed_R,True)
                    #test_model.update_seed_T(seed_T,True)
                    
                    # generate seed T
                    greedy_T_nodes = algor.MinGreedy(test_model,k,seed_R)
                    pagerank_T_nodes = algor.pagerank(G,k)
                    contrid_T_nodes = algor.ContrId(test_model,k,seed_R)
                    
                    # run diffusion function respectively

                    # greedy
                    test_model.update_seed_T(greedy_T_nodes,True)
                    greedy_num += len(test_model.diffusion()[2])
                    # pagerank
                    test_model.update_seed_T(pagerank_T_nodes,True)
                    pageranK_num += len(test_model.diffusion()[2])
                    # ContrId
                    test_model.update_seed_T(contrid_T_nodes,True)
                    contrid_num += len(test_model.diffusion()[2])

                    qbar.update(1)

            greedy_diffu_res.append(greedy_num/100)
            pagerank_diffu_res.append(pageranK_num/100)
            contrid_diffu_res.append(contrid_num/100)

    return greedy_diffu_res,pagerank_diffu_res,contrid_diffu_res


#%%
'''--------- unfinished ------------'''
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
'''---------- Run below -------------'''
calc_avg_R_diffu_num(G_scale_free,100,3,)
# %%
