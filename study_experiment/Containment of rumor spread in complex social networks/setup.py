#%%
from typing import Callable
import algorithm as algor
from LTD1DT import model_V2
import networkx as nx
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
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

    # select by random
    if rand_choose_R:
        df_node = DataFrame(G.nodes())
        
        for k in range(0,11):# select k(from 0 to 10) nodes of Truth
            print(f'choose {k} Truth nodes...')
            diffu_num = 0

            with tqdm(total=run_times) as qbar:
                for i in range(run_times): # run 100 times
                    # generate seed R
                    rand_seed_R = df_node.sample(seed_R_num,replace=False).to_numpy(dtype=int)
                    seed_R = np.reshape(rand_seed_R,rand_seed_R.shape[0])
                    test_model = model_V2(G,False,[],seed_R)
                    
                    # generate seed T
                    if len(fargs) == 0:
                        seed_T = func(test_model,seed_R,k)
                    else:
                        seed_T = func(G,seed_R,k,*fargs)

                    test_model.update_seed_T(seed_T,True)
                    diffu_num+= len(test_model.diffusion()[2])
                    qbar.update(1)

            diffu_result_list.append(diffu_num/run_times)
        return diffu_result_list
    
    # select by degree
    else:
        sorted_node_deg = sorted(list(nx.degree(G)),key=lambda x:x[1],reverse=True)
        seed_R = []
        for i in range(seed_R_num):
            seed_R.append(sorted_node_deg[i][0])

        for k in range(0,11):# select k(from 0 to 10) nodes of Truth
            print(f'choose {k} Truth nodes...')
            diffu_num = 0

            with tqdm(total=run_times) as qbar:
                for i in range(run_times):
                    # generate seed R

                    test_model = model_V2(G,False,[],seed_R)

                    
                    # generate seed T
                    if len(fargs) == 0:
                        seed_T = func(test_model,seed_R,k)
                    else:
                        seed_T = func(G,seed_R,k,*fargs)

                    test_model.update_seed_T(seed_T,True)
                    diffu_num+=len(test_model.diffusion()[2])

                    qbar.update(1)
            diffu_result_list.append(diffu_num/run_times)

        return diffu_result_list

#%%
def calc_avg_R_diffu_num_V2(G:nx.Graph,run_times:int=100,seed_R_num:int=3,random:bool=True):

    greedy_r = []
    greedy_t = []
    pagerank_r = []
    pagerank_t = []
    contrid_r = []
    contrid_t = []

    if random:
        df_node = DataFrame(G.nodes())
        pg = algor.pagerank(G,[],-1)
        for k in range(0,11):
            print(f'choose {k} Truth nodes...')
            greedy_num_r,pg_num_r,contrid_num_r =0,0,0
            greedy_num_t,pg_num_t,contrid_num_t =0,0,0

            with tqdm(total=run_times) as qbar:
                for i in range(run_times):

                    rand_seed_R = df_node.sample(seed_R_num,replace=False).to_numpy(dtype=int)
                    seed_R = np.reshape(rand_seed_R,rand_seed_R.shape[0])
                    test_model = model_V2(G,False,[],seed_R)

                    greedy_T = algor.MinGreedy(test_model,seed_R,k)
                    pg_T = algor.pagerank(G,seed_R,k)
                    contrid_T = algor.ContrId(test_model,seed_R,k)

                    test_model.update_seed_T(greedy_T,True)
                    res = test_model.diffusion()
                    greedy_num_r += len(res[2])
                    greedy_num_t += len(res[3])

                    test_model.update_seed_T(pg_T,True)
                    res = test_model.diffusion()
                    pg_num_r += len(res[2])
                    pg_num_t += len(res[3])

                    test_model.update_seed_T(contrid_T,True)
                    res = test_model.diffusion()
                    contrid_num_r += len(res[2])
                    contrid_num_t += len(res[3])

                    qbar.update(1)
            greedy_r.append(greedy_num_r/run_times)
            greedy_t.append(greedy_num_t/run_times)

            pagerank_r.append(pg_num_r/run_times)
            pagerank_t.append(pg_num_t/run_times)

            contrid_r.append(contrid_num_r/run_times)
            contrid_t.append(contrid_t/run_times)

        return {'greedy':[greedy_r,greedy_t],'pagerank':[pagerank_r,pagerank_t],'contrid':[contrid_r,contrid_t]}
    
    
    else:
        sorted_node_deg = sorted(list(nx.degree(G)),key=lambda x:x[1],reverse=True)
        seed_R = []
        for i in range(seed_R_num):
            seed_R.append(sorted_node_deg[i][0])

        pg = algor.pagerank(G,seed_R,-1)

        for k in range(0,11):
            print(f'choose {k} Truth nodes...')
            greedy_num_r,pg_num_r,contrid_num_r = 0,0,0
            greedy_num_t,pg_num_t,contrid_num_t =0,0,0
            pg_T = pg[:k]

            with tqdm(total=run_times) as qbar:
                for i in range(run_times):
                    test_model = model_V2(G,False,[],seed_R)

                    greedy_T = algor.MinGreedy(test_model,seed_R,k)

                    contrid_T = algor.ContrId(test_model,seed_R,k)

                    test_model.update_seed_T(greedy_T,True)
                    res = test_model.diffusion()
                    greedy_num_r += len(res[2])
                    greedy_num_t += len(res[3])

                    test_model.update_seed_T(pg_T,True)
                    res = test_model.diffusion()
                    pg_num_r += len(res[2])
                    pg_num_t += len(res[3])

                    test_model.update_seed_T(contrid_T,True)
                    res = test_model.diffusion()
                    contrid_num_r += len(res[2])
                    contrid_num_t += len(res[3])

                    qbar.update(1)
            greedy_r.append(greedy_num_r/run_times)
            greedy_t.append(greedy_num_t/run_times)

            pagerank_r.append(pg_num_r/run_times)
            pagerank_t.append(pg_num_t/run_times)

            contrid_r.append(contrid_num_r/run_times)
            contrid_t.append(contrid_t/run_times)

        return {'greedy':[greedy_r,greedy_t],'pagerank':[pagerank_r,pagerank_t],'contrid':[contrid_r,contrid_t]}
#%%
# def run_pagerank(G:nx.Graph,choose_T_num:int=0,seed_R_num:int=3,random:bool=True,fargs:tuple=()):

#     diffu_result = []
#     model = model_V2(G,False,[],[])
    
#     if random:
#         df_node = pd.DataFrame(G.nodes())
#         for i in range(choose_T_num):
#             seed_T = algor.pagerank(G,i,*fargs)
#             rand_seed_R = df_node.sample(seed_R_num,replace=False).to_numpy(dtype=int)
#             seed_R = np.reshape(rand_seed_R,rand_seed_R.shape[0])

#             model.update_seed_R(seed_R,True)
#             model.update_seed_T(seed_T,True)
#             diffu_result.append(len(model.diffusion()[2]))
#     else:
#         sorted_node_deg = sorted(list(nx.degree(G)),key=lambda x:x[1],reverse=True)
#         seed_R = []
#         for i in range(seed_R_num):
#             seed_R.append(sorted_node_deg[i][0])
#         model.update_seed_R(seed_R,True)

#         for i in range(choose_T_num):
#             seed_T = algor.pagerank(G,i,*fargs)
#             model.update_seed_T(seed_T,True)
#             diffu_result.append(len(model.diffusion()[2]))
#     return diffu_result
#%%
'''---------- Run below -------------'''
# sf_contrid = calc_avg_R_diffu_num(G_scale_free,algor.ContrId,rand_choose_R=False)
# # %%
# sf_greedy = calc_avg_R_diffu_num(G_scale_free,algor.MinGreedy,rand_choose_R=False)
# # %%
# sf_pg = calc_avg_R_diffu_num(G_scale_free,algor.pagerank,(0.85,),rand_choose_R=False)
# #%%
# sf_contrid
#%%
if __name__ == "__main__":
    import pickle
    #%%
    res_sf = calc_avg_R_diffu_num_V2(G_scale_free,random=False)
    with open('res_sf.pkl','wb') as file:
        pickle.dump(res_sf,file)
    #%%
    res_sw = calc_avg_R_diffu_num_V2(G_small_world,random=False)
    with open('res_sw.pkl','wb') as file:
        pickle.dump(res_sw,file)
    #%%
    res_nts = calc_avg_R_diffu_num_V2(G_netscience,random=False)
    with open('res_nts.pkl','wb') as file:
        pickle.dump(res_nts,file)
    #%%
    # res_us = calc_avg_R_diffu_num_V2(G_uspower,random=False)
    # with open('res_us.pkl','wb') as file:
    #     pickle.dump(res_us,file)

# %%
