#%%
import networkx as nx
from model import *
import math
#%%
'''
select other T nodes to restrict and correct the rumor nodes after detected the rumor
'''
#%%
# Containment of rumor spread in complex social networks
def ContrId_original(model:model, seed_R: list = None):
    '''Contributors Identification.
    The result has already excepted the seed of rumor node and authoritative T nodes.

    Parameters:
    ---------------
    model: model
        Just a model.
    seed_R: list
        The seed nodes of rumor
    '''
    if seed_R is None:
        print('please give a seed_R')
        return []
    else:
        res1 = model.unblocking_diffusion(seed_R,)

    node_contr = {}

    for node in nx.nodes(res1[0]):
        if node in seed_R:
            continue
        if node in model.authoritative_T:
            continue
        if res1[0].nodes[node]['group'] == 0:
            node_contr[node]=0
        
        elif res1[0].nodes[node]['group'] == 1:
            contr = 0
            for nbr in nx.neighbors(res1[0],node):
                if res1[0].nodes[nbr]['group'] == 1:
                    if res1[0].nodes[node]['active_time'] < res1[0].nodes[nbr]['active_time']:
                        contr += 1
            node_contr[node] = contr
    
    sorted_node_contr = sorted(node_contr.items(),key=lambda x:x[1],reverse=True)
    result = []
    for item in sorted_node_contr:
        result.append(item[0])
    
    return result

#----------------------------------------------------------------------------------------

# %%
# Minmizing rumor influence in multiplex online social networks based on human individual and social behaviors
def TCS(G:nx.Graph, spread_time:int, final_T_receiver:dict, final_R_receiver:dict, R_t_receiver_num:dict,k:int=0):

    f_VB_negative = 0
    # selected_T = {}
    T_nodes = []

    for i in range(k):
        
        if len(T_nodes) ==0:
            curr_largest_f_VB = 0
        else:
            curr_largest_f_VB = f_VB_negative

        candidate_node = -1

        for node in nx.nodes(G):
            if node in final_T_receiver:
                continue
            elif node in final_R_receiver:
                continue
            # elif node in selected_T:
            #     continue
            else:
            # calculate the node probability density
                sum_p_acc = 0
                cum_prod_e = 1
                f_v = 0
                
                # calculate fv(t)
                # for nbr in nx.neighbors(G,node):
                #     nbr_deg = nx.degree(G,nbr)
                #     nbr_nbr_R_active = 0
                #     for nbr_nbr in nx.neighbors(G,nbr):
                #         if G.nodes[nbr_nbr]['group'] == 1:
                #             nbr_nbr_R_active += 1
                #     sum_p_acc += nbr_nbr_R_active/nbr_deg
                #     if cum_prod_e == 0:
                #         cum_prod_e = np.exp(-(nbr_nbr_R_active/nbr_deg)*spread_time)
                #     else:
                #         cum_prod_e = cum_prod_e*np.exp(-(nbr_nbr_R_active/nbr_deg)*spread_time)
                # f_v = sum_p_acc*cum_prod_e

                # # the result of cumulative product of f_VB will be smaller and smaller in my model
                # # so I change to cumulative sum
                # if f_VB_negative+f_v > curr_largest_f_VB:
                #     curr_largest_f_VB = f_VB_negative+f_v
                #     candidate_node = node

                for nbr in nx.neighbors(G,node):
                    sum_p_acc += G.nodes[nbr]['i_threshold']
                    cum_prod_e *= math.exp(-(G.nodes[nbr]['i_threshold'])*spread_time)
                f_v = sum_p_acc * cum_prod_e
                if f_VB_negative+f_v>curr_largest_f_VB:
                    curr_largest_f_VB = f_VB_negative+f_v
                    candidate_node = node

        f_VB_negative = curr_largest_f_VB
        T_nodes.append(candidate_node)
        # selected_T[candidate_node]=f_VB_negative

    return T_nodes
#%%
#---------------------------------------
def degree_based(G:nx.Graph,final_T_receiver:dict, final_R_receiver:dict,k:int=0):
    sorted_deg = sorted(dict(G.degree()).items(),key=lambda x:x[1],reverse=True)
    result = []
    for node in sorted_deg:
        if node[0] in final_T_receiver:
            continue
        elif node[0] in final_R_receiver:
            continue
        else:
            result.append(node[0])
            k-=1
        if k<=0:
            break
    
    return result
# %%
'''
set monitoring T nodes to detect the rumor as soon as possible
'''
#%%
