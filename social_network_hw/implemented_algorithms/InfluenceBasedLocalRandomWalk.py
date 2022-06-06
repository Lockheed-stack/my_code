#%%
import networkx as nx
import time
import cupy as cp
import scipy.stats
import numpy as np
#%%
def create_transition_probability_matrix(G,num_of_node,node_influence_power):
    '''return a transition probability matrix.
    The matrix's shape is (G.max_num_of_node+1,G.max_num_of_node+1), 
    because there is no "node 0" or may miss some nodes
    in the dataset of "scholat link prediction".
    '''
    t_p_m = np.zeros((num_of_node+1,num_of_node+1))
    
    for node_x in nx.nodes(G):
        sum_of_influence = 0
        for nbr in nx.neighbors(G,node_x):
            sum_of_influence=np.add(sum_of_influence,node_influence_power[nbr])
        
        for nbr in nx.neighbors(G,node_x): 
            if sum_of_influence==0:
                t_p_m[node_x][nbr]=0
            else:
                t_p_m[node_x][nbr]=np.divide(node_influence_power[nbr],sum_of_influence)
            
    return t_p_m
#%%
def local_random_walk(time_steps,start_node,num_of_nodes,trans_prob_matrix):
    '''
    return a probability list after t steps
    '''
    t_step_output = cp.zeros((1,num_of_nodes+1))
    t_step_output[0][start_node]=1
    
    for steps in range(time_steps):
        t_step_output=cp.dot(t_step_output,trans_prob_matrix)

        
    return t_step_output
#%%
def nbr_influence_power(G):
    node_influence_power={}
    for node in nx.nodes(G):
        nbrs_deg = []
        all_neighbors_deg = 0
        
        for nbr in nx.neighbors(G,node):
            all_neighbors_deg+=nx.degree(G,nbr)
        
        for nbr in nx.neighbors(G,node):
            nbrs_deg.append(nx.degree(G,nbr)/all_neighbors_deg)
            
        node_influence_power[node]=scipy.stats.entropy(nbrs_deg)
    
    return node_influence_power
#%%
def score_LRW(G,time_steps,start_node):
    '''
    return a sorted dictionary of potential link for start node
    '''
    double_E = nx.number_of_edges(G)*2
    num_of_nodes = max(nx.nodes(G))
    score_dict={}
    nbr_influ = nbr_influence_power(G)
    trans_prob_matrix = create_transition_probability_matrix(G,num_of_nodes,nbr_influ)
    cp_trans_prob_matrix = cp.asarray(trans_prob_matrix)
    t_steps_output = local_random_walk(time_steps,start_node,num_of_nodes,cp_trans_prob_matrix)
    
    curr_progress = 0
    start_time = time.perf_counter()
    
    for node in nx.nodes(G):
        curr_progress+=1
        
        
        if node != start_node:
            if node not in nx.neighbors(G,start_node):
                start_node_deg = nx.degree(G,start_node)
                start_node_deg_div_2E = cp.divide(start_node_deg,double_E)
                startNode_to_node =cp.multiply(start_node_deg_div_2E,t_steps_output[0][node])
                
                tempOutput = local_random_walk(time_steps,node,num_of_nodes,cp_trans_prob_matrix)
                node_start_deg = nx.degree(G,node)
                node_start_deg_div_2E = cp.divide(node_start_deg,double_E)
                node_to_startNode =cp.multiply(node_start_deg_div_2E,tempOutput[0][start_node])
                
                score_dict[node]=cp.add(startNode_to_node,node_to_startNode)
        
        progress_bar(curr_progress,len(nx.nodes(G)),start_time)
   
    sorted_score_dict = sorted(score_dict.items(),key=lambda kv:kv[1],reverse=True)
    
    return sorted_score_dict
#%%
def progress_bar(cur_progress, total_progress, start_time):
    scale = 50  # 进度条长度
    cur_progress += 1  # 读出的excel表第一行为表头，第二行从0开始计数
    time_now = time.perf_counter()
    rate = total_progress/scale  # 总进度太长或太短，需要缩短或放大
    if rate == 0:  # 防止出错
        print('{:.0%}[{}->{}]{:.2f}s'.format(1, 50, 0, (time_now-start_time)))
        return

    completed = '*'*(int(cur_progress/rate))
    uncompleted = '-'*int((total_progress-cur_progress)/rate)
    percent = (cur_progress/total_progress)
    if percent >= 1:
        percent = 1
    print('\r{:^.0%}[{}->{}]{:.2f}s'.format(percent,
                                            completed, uncompleted, (time_now-start_time)), end='')
    # if (cur_progress) >= total_progress:
    #     print('')

# %%
if __name__=='__main__':
    G = nx.read_adjlist('./SCHOLAT Link Prediction/train.csv',nodetype=int,delimiter=',')
    output = score_LRW(G,5,8777)
#%%
output