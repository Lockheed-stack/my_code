#%%
import cupy as cp
import numpy as np
import time
from numba import cuda
from numba import jit
import networkx as nx
#%%
## Numpy and CPU
s = time.time()
x_cpu = np.ones((1000,1000,100))
e = time.time()
print(e - s)
### CuPy and GPU
s = time.time()
x_gpu = cp.ones((1000,1000,100))
e = time.time()
print(e - s)
# %%
s = time.time()
x_cpu *= 5
e = time.time()
print(e - s)
### CuPy and GPU
s = time.time()
x_gpu *= 5
e = time.time()
print(e - s)
# %%
# 不使用numba的情况
def t():
    x = 0
    for i in np.arange(500000):
        x += i
    return x
s = time.time()
t()
e = time.time()
print(e - s)
# %%
# 使用numba的情况
#@jit(nopython=True) 
def tt():
    x = 0
    for i in range(500000):
        x += i
    return x
s = time.time()
tt()
e = time.time()
print(e - s)
# %%
print(cuda.gpus)
# %%
# 高维矩阵/数组：
gpu = cp.ones( (1024,512,4,4) )
cpu = np.ones( (1024,512,4,4) )

# 纯numpy的cpu测试：
ctime1 = time.time()
for c in range(1024):
    cpu = np.add(cpu,cpu)   # 这里用np.add()和直接用 + 一样！内核都是cpu来算
ctime2 = time.time()
ctotal = ctime2 - ctime1
print('纯cpu计算时间：', ctotal)

# 纯cupy的gpu测试：
gtime1 = time.time()
for g in range(1024):
    gpu = cp.add(gpu,gpu)   # 自带的加法函数
gtime2 = time.time()
gtotal = gtime2 - gtime1
print('纯gpu计算时间：', gtotal)

# gpu和cpu混合编程：
ggtime1 = time.time()
for g in range(1024):
    gpu = gpu + gpu         # 手工加法：+ 默认回到cpu计算！！！
ggtime2 = time.time()
ggtotal = ggtime2 - ggtime1
print('混合的计算时间：', ggtotal)
# %%
G = nx.read_adjlist('./SCHOLAT Link Prediction/train.csv',nodetype=int,delimiter=',')
# %%
def create_transition_probability_matrix(G):
    '''return a transition probability matrix.
    The matrix's shape is (G.max_num_of_node+1,G.max_num_of_node+1), 
    because there is no "node 0" or may miss some nodes
    in the dataset of "scholat link prediction".
    '''
    size_num = max(nx.nodes(G))
    t_p_m = cp.zeros((size_num+1,size_num+1))
    
    for node_x in nx.nodes(G):
        for nbr in nx.neighbors(G,node_x):
            t_p_m[node_x][nbr]=1/nx.degree(G,node_x)
    
    return t_p_m

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
def score_LRW(G,time_steps,start_node):
    '''
    return a sorted dictionary of potential link for start node
    '''
    double_E = nx.number_of_edges(G)*2
    num_of_nodes = max(nx.nodes(G))
    score_dict={}
    trans_prob_matrix = create_transition_probability_matrix(G)
    t_steps_output = local_random_walk(time_steps,start_node,num_of_nodes,trans_prob_matrix)
    curr_progress = 0
    start_time = time.perf_counter()
    
    for node in nx.nodes(G):
        curr_progress+=1
        progress_bar(curr_progress,len(nx.nodes(G)),start_time)
        
        if node != start_node:
            if node not in nx.neighbors(G,start_node):
                startNode_to_node = nx.degree(G,start_node)/(double_E)*t_steps_output[node]
                
                tempOutput = local_random_walk(time_steps,node,num_of_nodes,trans_prob_matrix)
                node_to_startNode = nx.degree(G,node)/(double_E)*tempOutput[start_node]
                
                score_dict[node]=startNode_to_node+node_to_startNode
    
    sorted_score_dict = sorted(score_dict.items(),key=lambda kv:kv[1],reverse=True)
    
    return sorted_score_dict    
# %%
# %%
def progress_bar(cur_progress, total_progress, start_time):
    scale = 50  # 进度条长度
    cur_progress += 1  # 读出的excel表第一行为表头，第二行从0开始计数
    time_now = time.perf_counter()
    rate = total_progress/scale  # 总进度太长或太短，需要缩短或放大
    # if rate == 0:  # 防止出错
    #     print('{:.0%}[{}->{}]{:.2f}s'.format(1, 50, 0, (time_now-start_time)))
    #     return

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
output=score_LRW(G,5,126)
# %%
type(output[0][1])
# %%
gtime1 = time.time()
gpu = cp.array([1.3568956])
for g in range(102400):
    gpu = cp.add(gpu,gpu)   # 自带的加法函数
gtime2 = time.time()
gtotal = gtime2 - gtime1
print('纯gpu计算时间：', gtotal)
# %%
ctime1 = time.time()
cpu=np.array([1.3215448])
for c in range(102400):
    cpu = np.add(cpu,cpu)   # 这里用np.add()和直接用 + 一样！内核都是cpu来算
ctime2 = time.time()
ctotal = ctime2 - ctime1
print('纯cpu计算时间：', ctotal)
# %%
ggtime1 = time.time()
for g in range(102400):
    gpu = gpu + gpu         # 手工加法：+ 默认回到cpu计算！！！
ggtime2 = time.time()
ggtotal = ggtime2 - ggtime1
print('混合的计算时间：', ggtotal)
# %%
type(gpu)
# %%
a=np.array(1)
b=np.array(2)
a+b
# %%
c=cp.array((6),dtype=np.float64)
#d = cp.array((478.44556522))
c
# %%

# %%
