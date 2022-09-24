#%%
from typing import Callable
import algorithm as algor
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
def calc_avg_R_diffu_num(G:nx.Graph,func:Callable,fargs:tuple=(),run_times:int=100,seed_R_num:int=3,rand_choose_R:bool=True):
    ''' Using three different algorithms respectively to calculate the number of rumor diffusion.

    Parameters
    ----------
    G : nx.Graph
        A Networkx graph
    func : Callable
        Specify a algorithm function to running.
    fargs : tuple, optional
        The argument passed to the function.(No need for now.)
    run_times : int, optional
        The number of simulation of diffusion times, by default 100
    seed_R_num : int, optional
        The rumor nodes number, by default 3
    rand_choose_R : bool, optional
        Decide whether to use random seed_R , by default True

    Returns
    -------
    diffu_result_list : list
        Return a list of diffusion result which select Truth nodes from 1 to 10 respectively.
    '''
    diffu_result_list = []
    # greedy_diffu_res = []
    # pagerank_diffu_res = []
    # contrid_diffu_res = []

    # select by random
    if rand_choose_R:
        df_node = pd.DataFrame(G.nodes())
        
        for k in range(0,11):# select k(from 0 to 10) nodes of Truth
            print(f'choose {k} Truth nodes...')
            diffu_num = 0

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
                    if len(fargs) == 0:
                        seed_T = func(test_model,seed_R,k)
                    else:
                        seed_T = func(G,seed_R,k,*fargs)
                    # greedy_T_nodes = algor.MinGreedy(test_model,k,seed_R)
                    # pagerank_T_nodes = algor.pagerank(G,k)
                    # contrid_T_nodes = algor.ContrId(test_model,k,seed_R)
                    
                    # run diffusion function respectively

                    # greedy
                    # test_model.update_seed_T(greedy_T_nodes,True)
                    # greedy_num += len(test_model.diffusion()[2])
                    # # pagerank
                    # test_model.update_seed_T(pagerank_T_nodes,True)
                    # pageranK_num += len(test_model.diffusion()[2])
                    # # ContrId
                    # test_model.update_seed_T(contrid_T_nodes,True)
                    # contrid_num += len(test_model.diffusion()[2])
                    test_model.update_seed_T(seed_T,True)
                    diffu_num+= len(test_model.diffusion()[2])
                    qbar.update(1)

            # greedy_diffu_res.append(greedy_num/100)
            # pagerank_diffu_res.append(pageranK_num/100)
            # contrid_diffu_res.append(contrid_num/100)
            diffu_result_list.append(diffu_num/run_times)
        return diffu_result_list
        # return greedy_diffu_res,pagerank_diffu_res,contrid_diffu_res
    
    
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
            # greedy_num,pageranK_num,contrid_num=0,0,0
            diffu_num = 0

            with tqdm(total=run_times) as qbar:
                for i in range(run_times):
                    # generate seed R

                    test_model = model_V2(G,False,[],seed_R)
                    #test_model.update_seed_R(seed_R,True)
                    #test_model.update_seed_T(seed_T,True)
                    
                    # generate seed T
                    if len(fargs) == 0:
                        seed_T = func(test_model,seed_R,k)
                    else:
                        seed_T = func(G,seed_R,k,*fargs)
                    # greedy_T_nodes = algor.MinGreedy(test_model,k,seed_R)
                    # pagerank_T_nodes = algor.pagerank(G,k)
                    # contrid_T_nodes = algor.ContrId(test_model,k,seed_R)
                    
                    # run diffusion function respectively

                    # greedy
                    # test_model.update_seed_T(greedy_T_nodes,True)
                    # greedy_num += len(test_model.diffusion()[2])
                    # # pagerank
                    # test_model.update_seed_T(pagerank_T_nodes,True)
                    # pageranK_num += len(test_model.diffusion()[2])
                    # # ContrId
                    # test_model.update_seed_T(contrid_T_nodes,True)
                    # contrid_num += len(test_model.diffusion()[2])
                    test_model.update_seed_T(seed_T,True)
                    diffu_num+=len(test_model.diffusion()[2])

                    qbar.update(1)
            diffu_result_list.append(diffu_num/run_times)
            # greedy_diffu_res.append(greedy_num/100)
            # pagerank_diffu_res.append(pageranK_num/100)
            # contrid_diffu_res.append(contrid_num/100)
        return diffu_result_list
        # return greedy_diffu_res,pagerank_diffu_res,contrid_diffu_res


#%%
def run_pagerank(G:nx.Graph,choose_T_num:int=0,seed_R_num:int=3,random:bool=True,fargs:tuple=()):

    diffu_result = []
    model = model_V2(G,False,[],[])
    
    if random:
        df_node = pd.DataFrame(G.nodes())
        for i in range(choose_T_num):
            seed_T = algor.pagerank(G,i,*fargs)
            rand_seed_R = df_node.sample(seed_R_num,replace=False).to_numpy(dtype=int)
            seed_R = np.reshape(rand_seed_R,rand_seed_R.shape[0])

            model.update_seed_R(seed_R,True)
            model.update_seed_T(seed_T,True)
            diffu_result.append(len(model.diffusion()[2]))
    else:
        sorted_node_deg = sorted(list(nx.degree(G)),key=lambda x:x[1],reverse=True)
        seed_R = []
        for i in range(seed_R_num):
            seed_R.append(sorted_node_deg[i][0])
        model.update_seed_R(seed_R,True)

        for i in range(choose_T_num):
            seed_T = algor.pagerank(G,i,*fargs)
            model.update_seed_T(seed_T,True)
            diffu_result.append(len(model.diffusion()[2]))
    return diffu_result
#%%
'''---------- Run below -------------'''
sf_contrid = calc_avg_R_diffu_num(G_scale_free,algor.ContrId,rand_choose_R=False)
# %%
sf_greedy = calc_avg_R_diffu_num(G_scale_free,algor.MinGreedy,rand_choose_R=False)
# %%
sf_pg = calc_avg_R_diffu_num(G_scale_free,algor.pagerank,(0.85,),rand_choose_R=False)
#%%
sf_greedy
# %%
sf_contrid
# %%
sf_pg

