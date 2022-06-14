#%%
import torch
#import numpy as np
import networkx as nx
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
def local_random_walk(time_steps,start_node,num_of_nodes,trans_prob_matrix):
    '''
    return a probability list after t steps
    '''
    #final_output_dict={}
    dev = torch.device('cuda')
    t_step_output = torch.zeros((1,num_of_nodes+1),device=dev)
    t_step_output[0][start_node]=1
    
    for steps in range(time_steps):
        #t_step_output=cp.dot(t_step_output,trans_prob_matrix)
        torch.mm(t_step_output,trans_prob_matrix,out=t_step_output)
    # for i in range(1,max(nx.nodes(G))+1):
    #     final_output_dict[i]=t_step_output[i]
        
    return t_step_output
# %%
def score_LRW(G,time_steps,start_node):
    '''
    return a sorted dictionary of potential link for start node
    '''
    double_E = nx.number_of_edges(G)*2
    num_of_nodes = max(nx.nodes(G))
    score_dict={}
    trans_prob_matrix = create_transition_probability_matrix(G)
    t_steps_output = local_random_walk(time_steps,start_node,num_of_nodes,trans_prob_matrix)
    # curr_progress = 0
    # start_time = time.perf_counter()
    with tqdm(total=len(nx.nodes(G))) as qbar:
        for node in nx.nodes(G):
            # curr_progress+=1
            # progress_bar(curr_progress,len(nx.nodes(G)),start_time)
            
            if node != start_node:
                if node not in nx.neighbors(G,start_node):
                    startNode_to_node = nx.degree(G,start_node)/(double_E)*t_steps_output[0][node]
                    
                    tempOutput = local_random_walk(time_steps,node,num_of_nodes,trans_prob_matrix)
                    node_to_startNode = nx.degree(G,node)/(double_E)*tempOutput[0][start_node]
                    
                    score_dict[node]=startNode_to_node+node_to_startNode
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
#                                             completed, uncompleted, (time_now-start_time)), end='',flush=True)
#     if (cur_progress) >= total_progress:
#         print(' ,Done')
# %%
# if __name__=='__main__':
#     G = nx.read_adjlist('../SCHOLAT Link Prediction/train.csv',nodetype=int,delimiter=',')
#     output=score_LRW(G,6,8777)
# # %%
# output
# # %%
# def AUC(output,G_test,node):
#     '''
#     @return auc score
#     @param:
#         @output: list-like, list[set(),set(),..,set()]
#         @G_test: the graph for test
#         @node: start node
#     '''
#     missing_edge={}
#     for i in output:
#         if i[0] in nx.neighbors(G_test,node):
#             missing_edge[i[0]]=i[1]
    
#     n1,n2 = 0,0
#     for edge in missing_edge.items():
#         for i in output:
#             if i[0] not in nx.neighbors(G_test,node):#non-existent edge
#                 if edge[1]<i[1]:
#                     continue
#                 elif edge[1]>i[1]:
#                     n1+=1
#                 else:
#                     n2+=1
#     n = len(missing_edge)*len(output)
#     auc = (n1+0.5*n2)/n
#     print(f'node {node}\'s AUC: {auc}')
#     return auc
# # %%
# G_test = nx.read_adjlist('../SCHOLAT Link Prediction/test.csv',nodetype=int,delimiter=',')
# AUC(output,G_test,8777)
# %%
