#%%
from model import *
import networkx as nx
from queue import SimpleQueue
from tqdm import tqdm
from multiprocessing import Process,Manager
import argparse
import os
import pickle
#%%
def run_separately(result,range_nodes:list,model:model,G:nx.Graph, spread_time:int, final_T_receiver:dict, final_R_receiver:dict, R_t_receiver_num:dict,T_nodes:list):
    
    delta = 1e6
    temp_node = None
    temp_res = None
    # with tqdm(total=len(range_nodes)) as qbar:
    for node in range_nodes:
        if node in final_R_receiver:
            continue
        if node in final_T_receiver:
            continue
        res = model.after_detected_diffusion(G,spread_time,final_T_receiver,final_R_receiver,R_t_receiver_num,T_nodes+[node])
        # print(res)
        if temp_res is None:
            temp_res = res
        if temp_node is None:
            temp_node = node

        if len(res[3])-len(res[2]) < delta:
            temp_node = node
            delta = len(res[3])-len(res[2])
            temp_res = res
                # qbar.update(1)
    
    result.put((delta,temp_node,temp_res))
    # print(f'subprocess:{os.getpid()}, done')

#%%
def greedy(process:int,model:model,G:nx.Graph, spread_time:int, final_T_receiver:dict, final_R_receiver:dict, R_t_receiver_num:dict,k:int=0):
    sub_result = Manager().Queue()
    choose_nodes = []
    
    nodes = list(G.nodes)
    range_size = int(len(nodes)/process)
    apart_nodes = []
    for i in range(process):
        if i+1 == process:
            apart_nodes.append(nodes[i*range_size:])
        else:
            apart_nodes.append(nodes[i*range_size:(i+1)*range_size])
    
    all_process = SimpleQueue()
    for i in range(process*k):
        all_process.put(Process(target=run_separately,args=(sub_result,apart_nodes[i%process],model,G,spread_time,final_T_receiver,final_R_receiver,R_t_receiver_num,choose_nodes)))

    temp_res = None
    temp_node = None
    with tqdm(total=process*k) as qbar:
        while not all_process.empty():        
            sub_process = []
            for i in range(process):
                p = all_process.get()
                p.start()
                sub_process.append(p)
            for proc in sub_process:
                proc.join()

            delta = 1e6
            while not sub_result.empty():
                result = sub_result.get()
                if temp_node is None:
                    temp_node = result[1]
                if temp_res is None:
                    temp_res = result[2]

                if result[0] < delta:
                    delta = result[0]
                    temp_node = result[1]
                    temp_res = result[2]
            choose_nodes.append(temp_node)
            final_T_receiver[temp_node]=1
            qbar.update(process)
    return temp_res
#%%
def sim_greedy(process:int,model:model,au_T_rate:float=0.005,r_rate:float=0.01,T_rate:float=0.01,iters:int=500):
    final_R_num_avg = 0.0
    final_T_num_avg = 0.0
    avg_spread_time = 0.0
    num_of_nodes = nx.number_of_nodes(model.G)

    select_num = int(num_of_nodes*T_rate)
    if select_num <1:
        select_num = 1
        

    for j in range(iters):
        if au_T_rate:
            model.refresh_i_c_threshold(True,au_T_rate)
        
        r_n = model.generate_R_nodes(0,r_rate)
        res1 = model.before_detected_diffusion(r_n)

        res2 = greedy(process,model,*res1,select_num)

        final_R_num_avg += res2[4][res2[1]-1]
        final_T_num_avg += len(res2[2])
        avg_spread_time += res2[1]
        print('iter: %d, R:%.3f, T:%.3f, s_t:%.3f'%(j+1,final_R_num_avg/(j+1),final_T_num_avg/(j+1),avg_spread_time/(j+1)))

    final_R_num_avg /= iters
    final_T_num_avg /= iters
    avg_spread_time /= iters

    return final_R_num_avg,final_T_num_avg,avg_spread_time,T_rate

#%%

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subprocess_num',type=int,help='processes num')
    parser.add_argument('--iters',type=int,help='total iters times')
    parser.add_argument('--T_rate',type=float,help='the percentage of truth node')
    parser.add_argument('--name',type=str,help='name of saving pickle',default=None)
    
    args = parser.parse_args()
    subprocess_num = args.subprocess_num
    iters=args.iters
    T_rate = args.T_rate
    name = args.name
    # subprocess_num = 4
    # iters=1
    # T_rate = 0.005

    G = nx.read_adjlist('../dataset/USpowerGrid.mtx',nodetype=int,)
    model1 = model(G,0.004,)
    
    res = sim_greedy(subprocess_num,model1,0.005,0.05,T_rate,iters)
    if name is None:
        with open(f'test4_{T_rate}_{iters}.pkl','wb') as f:
            pickle.dump(res,f)
    else:
        with open(f'{name}.pkl','wb') as f:
            pickle.dump(res,f)
    print(res)
 
# %%
