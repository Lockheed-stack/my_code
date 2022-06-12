#%%
import torch
#import numpy as np
import networkx as nx
#import cupy as cp
from tqdm import tqdm
# %%
def create_transition_probability_matrix(G):
    '''return a transition probability matrix.
    The matrix's shape is (G.max_num_of_node+1,G.max_num_of_node+1), 
    because there is no "node 0" or may miss some nodes
    in the dataset of "scholat link prediction".
    '''
    dev = torch.device('cuda')
    size_num = max(nx.nodes(G))
    t_p_m = torch.zeros((size_num+1,size_num+1),device=dev)
    
    for node_x in nx.nodes(G):
        for nbr in nx.neighbors(G,node_x):
            t_p_m[node_x][nbr]=1/nx.degree(G,node_x)
    
    return t_p_m

# %%
#@profile
def local_random_walk(time_steps,start_node,end_node,start_node_deg,end_node_deg,double_E,num_of_nodes,trans_prob_matrix):
    '''
    return a probability list after t steps
    '''
    #final_output_dict={}
    dev = torch.device('cuda')
    start_step_output = torch.zeros((1,num_of_nodes+1),device=dev)
    start_step_output[0][start_node]=1
    end_step_output = torch.zeros((1,num_of_nodes+1),device=dev)
    end_step_output[0][end_node]=1
    
    A_to_cuda = torch.div(start_node_deg,double_E).to('cuda')
    B_to_cuda = torch.div(end_node_deg,double_E).to('cuda')
    superpose_result = torch.tensor(0.0,device=dev)
    for steps in range(time_steps):
        torch.mm(start_step_output,trans_prob_matrix,out=start_step_output)
        torch.mm(end_step_output,trans_prob_matrix,out=end_step_output)
        start_to_end = torch.multiply(A_to_cuda,start_step_output[0][end_node]).to('cuda')
        end_to_start = torch.multiply(B_to_cuda,end_step_output[0][start_node]).to('cuda')
        torch.add(start_to_end,end_to_start,out=superpose_result).to('cuda')
    # for i in range(1,max(nx.nodes(G))+1):
    #     final_output_dict[i]=t_step_output[i]
        
    return superpose_result
# %%
#@profile
def score_SRW(G,time_steps,start_node):
    '''
    return a sorted dictionary of potential link for start node
    '''
    double_E = nx.number_of_edges(G)*2
    num_of_nodes = max(nx.nodes(G))
    score_dict={}
    trans_prob_matrix = create_transition_probability_matrix(G)
    #t_steps_output = local_random_walk(time_steps,start_node,num_of_nodes,trans_prob_matrix)
    #curr_progress = 0
    #start_time = time.perf_counter()
    with tqdm(total=len(nx.nodes(G))) as qbar:
        for node in nx.nodes(G):
            #curr_progress+=1
            
            
            if node != start_node:
                if node not in nx.neighbors(G,start_node):
                    # sum_of_LRW = 0
                    # for i in range(1,time_steps+1):
                    #     t_steps_output = local_random_walk(i,start_node,num_of_nodes,trans_prob_matrix)
                    #     startNode_to_node = nx.degree(G,start_node)/(double_E)*t_steps_output[0][node]
                        
                    #     tempOutput = local_random_walk(i,node,num_of_nodes,trans_prob_matrix)
                    #     node_to_startNode = nx.degree(G,node)/(double_E)*tempOutput[0][start_node]
                        
                    #     sum_of_LRW += startNode_to_node+node_to_startNode
                    
                    score_dict[node]=local_random_walk(time_steps,start_node,node,
                                                       nx.degree(G,start_node),nx.degree(G,node),double_E,
                                                       num_of_nodes,trans_prob_matrix)
            #progress_bar(curr_progress,len(nx.nodes(G)),start_time)
            qbar.update(1)
    sorted_score_dict = sorted(score_dict.items(),key=lambda kv:kv[1],reverse=True)
    
    return sorted_score_dict    
# %%
# %%
# def progress_bar(cur_progress, total_progress, start_time):
#     scale = 50  # 进度条长度
#     cur_progress += 1  # 读出的excel表第一行为表头，第二行从0开始计数
#     time_now = time.perf_counter()
#     rate = total_progress/scale  # 总进度太长或太短，需要缩短或放大
#     # if rate == 0:  # 防止出错
#     #     print('{:.0%}[{}->{}]{:.2f}s'.format(1, 50, 0, (time_now-start_time)))
#     #     return

#     completed = '*'*(int(cur_progress/rate))
#     uncompleted = '-'*int((total_progress-cur_progress)/rate)
#     percent = (cur_progress/total_progress)
#     if percent >= 1:
#         percent = 1
#     print('\r{:^.0%}[{}->{}]{:.2f}s'.format(percent,
#                                             completed, uncompleted, (time_now-start_time)), end='')
#     if (cur_progress) >= total_progress:
#         print(' ,Done')
# # %%
# if __name__=='__main__':
#     G = nx.read_adjlist('../SCHOLAT Link Prediction/train.csv',nodetype=int,delimiter=',')
#     output = score_SRW(G,5,8777)
# #%%    
# output
# # %%
# output
# %%
