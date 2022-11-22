#%%
import networkx as nx
from model import *
import math
from queue import SimpleQueue
from scipy.stats import entropy
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
    T_nodes = []
    checked_node = {}
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
            elif node in checked_node:
                continue
            else:
            # calculate the node probability density
                sum_p_acc = 0
                cum_prod_e = 1
                f_v = 0

                for nbr in nx.neighbors(G,node):
                    sum_p_acc += G.nodes[nbr]['i_threshold']
                    cum_prod_e *= math.exp(-(G.nodes[nbr]['i_threshold'])*spread_time)
                f_v = sum_p_acc * cum_prod_e
                if f_VB_negative+f_v>curr_largest_f_VB:
                    curr_largest_f_VB = f_VB_negative+f_v
                    candidate_node = node

        f_VB_negative = curr_largest_f_VB
        T_nodes.append(candidate_node)
        checked_node[candidate_node]=f_VB_negative

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
# ----------------------------------------------
#%%
# motif based 
def motif_3_trt(G:nx.Graph,final_T_receiver:dict,final_R_receiver:dict,k:int=0):

    result = [] # return value
    
    cover_R = {}

    T_around = {}
    
    candidate_node_queue = SimpleQueue()

# ---------------------- Using exiting T nodes to create motif-3 ------------------------------------
    # searching around R-nodes which were detected by T
    for node in final_R_receiver.keys():
        num_T_around = 0
        for nbr in nx.neighbors(G,node):
            if G.nodes[nbr]['group'] == 2: # T-active node
                num_T_around+=1
            elif G.nodes[nbr]['group'] == 0: # inactive node
                if num_T_around>0:
                    candidate_node_queue.put(nbr)
                    if nbr in cover_R:
                        cover_R[nbr]+=1 # covering more R-nodes is preferred
                    else:
                        cover_R[nbr]=1
        
        # The more T nodes surround R node is perferred
        while not candidate_node_queue.empty():
            c_node = candidate_node_queue.get()
            if c_node in T_around:
                T_around[c_node]+=num_T_around
            else:
                T_around[c_node]=num_T_around
    
    if k < len(cover_R): # the budget is less than the best scheme budget
        mid_term_result = {}
        for node in cover_R.keys():
            mid_term_result[node]=(cover_R[node],T_around[node])
        
        sorted_mid_term_result = sorted(mid_term_result.items(),key=lambda x:(x[1][0],x[1][1]),reverse=True)
        for node in sorted_mid_term_result:
            result.append(node[0])
            k-=1
            if k<=0:
                break
        return result
# ------------------- Creating motif-3 by using 2 inactive nodes ---------------------------
    else:
        for node in cover_R.keys():# put nodes from cover_R into result
            result.append(node)
        
        i_node_score = {}
        for node in final_R_receiver.keys():
            checked_node = {}
            for i_nbr in nx.neighbors(G,node):
                if i_nbr in cover_R:
                    continue
                elif i_nbr not in checked_node:
                    checked_node[i_nbr]=1
                    candidate_node_queue.put(i_nbr)
                
        while not candidate_node_queue.empty():
            i_node = candidate_node_queue.get()
            nbr_R_num = 0
            nbr_R_deg_sum = 0
            for nbr in nx.neighbors(G,i_node):
                if nbr in final_R_receiver.keys():
                    nbr_R_num += 1
                    nbr_R_deg_sum += G.degree(nbr)
            
            i_node_score[i_node] = G.degree(i_node)/(1+(nbr_R_deg_sum/nbr_R_num))
        
        if (k-len(result)) < len(i_node_score):
            select_num = k-len(result)
            for node in sorted(i_node_score.items(),key=lambda x:x[1],reverse=True):
                result.append(node[0])
                select_num-=1
                if select_num<=0:
                    break
            
            return result
        
        # still on budget
        else:
            for node in i_node_score.keys():
                result.append(node)
            # select T nodes based on degree
            select_num = k-len(result)
            for node in sorted(dict(G.degree()).items(),key=lambda x:x[1],reverse=True):
                if node[0] not in final_T_receiver:
                    if node[0] not in final_R_receiver:
                        if node[0] not in cover_R:
                            if node[0] not in i_node_score:
                                result.append(node[0])
                                select_num-=1
                                if select_num<=0:
                                    break
            return result
# ---------------------------------------------
# %%
'''
set monitoring T nodes to detect the rumor as soon as possible
'''
#%%
# degree based algorithm.
def set_M_degree(model:model,k:int=0):
    sorted_deg = sorted(dict(model.G.degree()).items(),key=lambda x:x[1],reverse=True)
    result = []

    if k==0:
        for node in sorted_deg:
            result.append(node[0])
    else:
        for node in sorted_deg:
            if node[0] in model.authoritative_T:
                continue
            else:
                result.append(node[0])
                k-=1
            if k<=0:
                break
    
    return result
#%%
# random select 
def set_M_random(model:model,k:int=0):
    return list(model.droped_auT_df.sample(k,replace=False).to_dict('index').keys())
# %%
# Identifying Important Nodes in Complex Networks Based on Node Propagation Entropy
def propagation_entropy(model:model,k:int=0):
    '''Using propagation entropy to select monitoring T nodes
        Note: The result has been removed the authoritative T nodes.

    Parameters
    ----------
    model : model
        my model
    k : int, optional
        the number of monitoring T nodes, by default 0.
        Note: If k=0 then will return the propagation entropy of all nodes.(except the authoritative T nodes)

    Returns
    -------
    list
        the list of propagation entropy of nodes.
    '''
    # compute the clustering coefficient: c_i
    c_i = nx.clustering(model.G,)
    
    # compute the clustering coefficient and neighbors: cn_i
    cn_i = {}
    for node in model.G.nodes():
        nbr_queue = SimpleQueue()
        check_nbr = {}
        n1_num = nx.degree(model.G,node)
        n2_num = 0
        for nbr in nx.neighbors(model.G,node):
            nbr_queue.put(nbr)
        
        while not nbr_queue.empty():
            get_node = nbr_queue.get()
            for nbr in nx.neighbors(model.G,get_node):
                if nbr not in check_nbr:
                    n2_num+=1
                    check_nbr[nbr]=1
        
        cn_i[node]=(n1_num+n2_num)/(c_i[node]+1)
    
    # compute the entropy: PE_i
    sum_cn_i = sum(cn_i.values())
    I_i = {}
    PE_i = {}

    for node in model.G.nodes():
        I_i[node] = cn_i[node]/sum_cn_i
    
    for node in model.G.nodes():
        nbr_I_i_list = []
        for nbr in nx.neighbors(model.G,node):
            nbr_I_i_list.append(I_i[nbr])
        PE_i[node] = entropy(nbr_I_i_list)
    
    # sorted
    sorted_PE = sorted(PE_i.items(),key=lambda x:x[1],reverse=True)
    result = []
    if k==0:
        for node in sorted_PE:
            result.append(node[0])
    else:
        for node in sorted_PE:
            if node[0] in model.authoritative_T:
                continue
            else:
                result.append(node[0])
                k-=1
                if k<=0:
                    break
    
    return result
# %%
# %%
