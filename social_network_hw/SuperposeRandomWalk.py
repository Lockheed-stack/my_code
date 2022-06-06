#%%
import networkx as nx
import time
import cupy as cp
from numpy import float64
#from numpy import zeros,dot
#from line_profiler import LineProfiler
#import pandas as pd
# %%
#@profile
def local_random_walk(time_steps,start_node,num_of_nodes,trans_prob_matrix):
    '''
    return a probability list after t steps
    '''
    #final_output_dict={}
    t_step_output = cp.zeros(num_of_nodes+1)
    t_step_output[start_node]=1
    
    for steps in range(time_steps):
        t_step_output=cp.dot(t_step_output,trans_prob_matrix)
        
    # for i in range(1,max(nx.nodes(G))+1):
    #     final_output_dict[i]=t_step_output[i]
        
    return t_step_output


#%%
def create_transition_probability_matrix(G,num_of_node):
    '''return a transition probability matrix.
    The matrix's shape is (G.max_num_of_node+1,G.max_num_of_node+1), 
    because there is no "node 0" or may miss some nodes
    in the dataset of "scholat link prediction".
    '''
    t_p_m = cp.zeros((num_of_node+1,num_of_node+1))
    
    for node_x in nx.nodes(G):
        for nbr in nx.neighbors(G,node_x):
            t_p_m[node_x][nbr]=1/nx.degree(G,node_x)
    
    return t_p_m
#%%
#@profile
def score_SRW(G,time_steps,start_node):
    double_E = nx.number_of_edges(G)*2
    num_of_nodes = max(nx.nodes(G))
    score_dict={}
    trans_prob_matrix = create_transition_probability_matrix(G,num_of_nodes)
    
    curr_progress = 0
    start_time = time.perf_counter()
    
    for node in nx.nodes(G):
        curr_progress+=1
        
        
        if node != start_node:
            if node not in nx.neighbors(G,start_node):
                score=cp.array(0,dtype=float64) # score=0
                
                for i in range(1,time_steps+1):
                    t_steps_output = local_random_walk(i,start_node,num_of_nodes,trans_prob_matrix)
                    startNode_to_node = nx.degree(G,start_node)/(double_E)*t_steps_output[node]
                    
                    tempOutput = local_random_walk(i,node,num_of_nodes,trans_prob_matrix)
                    node_to_startNode = nx.degree(G,node)/(double_E)*tempOutput[start_node]
                    score=cp.add(startNode_to_node,node_to_startNode)
                    
            score_dict[node]=score
            
        progress_bar(curr_progress,len(nx.nodes(G)),start_time)
        
    sorted_score = sorted(score_dict.items(),key=lambda kv:kv[1],reverse=True)
    
    return sorted_score 
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
    output = score_SRW(G,6,126)
    # output_file=pd.DataFrame(output)
    # output_file.to_csv('score_SRW.csv',index=False,header=False)
#%%
output
# %%
