#%%
from itertools import permutations
import networkx as nx
import create_spanning_graph as csg
import stenier_distance as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
# %%
def select_k_node(k:int,G:nx.Graph):
    res = np.array(csg.all_combination(k,G.number_of_nodes()))
    return res+1 # because the all_combination return a list which node's label start from zero
#%%

def assignment_stenier_value(tensor_G:np.ndarray,node_list:list,value:int,loc:int=0):
    if np.shape(tensor_G[node_list[loc]]) != ():
        assignment_stenier_value(tensor_G[node_list[loc]],node_list,value,loc+1)

    else:
        tensor_G[node_list[loc]]=value

    return
# %%
def create_Graph_tensor(k:int,G:nx.Graph):
    # make sure that k<=G.number_of_nodes
    if k>G.number_of_nodes():
        print('please make sure that k<=G.number_of_nodes')
        return 
    n = G.number_of_nodes()
    k_node_array = select_k_node(k,G)
    tensor_shape = tuple([n]*k)

    tensor_G = np.zeros(tensor_shape,dtype=int)

    for node_list in k_node_array:
        value = st.mini_steiner_tree(G,node_list)
        temp = node_list-1
        for permutations_node_list in list(permutations(temp)):
            assignment_stenier_value(tensor_G,permutations_node_list,value)

    return tensor_G

# %%
def tensor_eigenvalue2(tensor_G:np.ndarray):
    if tensor_G[0]:
        pass
#%%
def tensor_eigenvalue(tensor_G:np.ndarray):
    ndim = tensor_G.shape[0]
    x = np.ones(ndim)
    y = np.zeros(ndim)
    t = 10
    iter = 1
    while t>=0.0001:
        
        for i in range(ndim):
            s = 0
            for j in range(ndim):
                for k in range(ndim):
                    s+=tensor_G[i,j,k]*x[j]*x[k]
            y[i]=s

        min_value = min(y/(x**2))
        max_value = max(y/(x**2))

        t = np.abs(max_value-min_value)
        x = (np.sqrt(y))/np.sqrt(sum(y))
        iter+=1
        if iter>=1000:
            break
    return (min_value+max_value)/2
#%%
def start(k:int,node_num:int,threshold:int,subgraph:dict):
    G = nx.complete_graph(range(1,node_num+1))
    #G_subgraph = csg.create_spanning_subgraph(G)
    G_subgraph = subgraph
    tensor_G = create_Graph_tensor(k,G)
    G_tensor_eigenvalue = tensor_eigenvalue(tensor_G)
    same_eigenvalue_G = []
    for item in G_subgraph.items():
        if item[0] == []:
            continue
        for sub_G in item[1]:
            tensor_sub = create_Graph_tensor(k,sub_G)
            sub_tensor_eigenvalue = tensor_eigenvalue(tensor_sub)
            if np.abs(sub_tensor_eigenvalue - G_tensor_eigenvalue)<=threshold:
                same_eigenvalue_G.append({sub_G:sub_tensor_eigenvalue})

    return same_eigenvalue_G,G_tensor_eigenvalue

#%%
def draw_result(ans:list,original_G:nx.Graph,original_eig_v:float,k:int):
    num = len(ans)
    if num == 0:
        print('No result')
        return 
    
    # if num<=11:
    #     figure,axs = plt.subplots(1,num+1,figsize=(int(10*(np.log10(num)+1)), 6),constrained_layout=True)
    # else:
    nrow = int(np.sqrt(num))+1
    figure,axs = plt.subplots(nrow,nrow,
                                figsize=(int(10*(np.log10(num)+1)),(int(10*(np.log10(num)+1)))),
                                constrained_layout=True)

    nx.draw(original_G,ax=axs[0,0],pos=nx.circular_layout(original_G),with_labels=True,font_color='w',font_size=13)
    axs[0,0].set_title('Complete Graph',fontsize=15)
    
    i=0
    # for G_dict in ans:
    #     G = list(G_dict.items())[0][0]
    #     eig_v = list(G_dict.items())[0][1]
    #     nx.draw(G,ax=axs[i],pos=nx.circular_layout(G)
    #     ,with_labels=True,font_color='w',font_size=13)
    #     axs[i].set_title(f'eigenvalue:{eig_v:.3f}')
    #     i+=1
    j=0
    for ax_out in axs:
        for ax_loc in ax_out:
            if j==0:
                j=1
                continue
            if i<len(ans):
                G = list(ans[i].items())[0][0]
                eig_v = list(ans[i].items())[0][1]
                nx.draw(G,ax=ax_loc,pos=nx.circular_layout(G)
                        ,with_labels=True,font_color='w',font_size=13)
                ax_loc.set_title(f'eigenvalue:{eig_v:.3f}')
                i+=1
            else:
                ax_loc.axis('off')

    plt.suptitle(f'Tensor eigenvalue:{original_eig_v},K={k},N={nx.number_of_nodes(original_G)}',fontsize=26)
    plt.savefig(f'k={k}_n={nx.number_of_nodes(original_G)}.jpg',bbox_inches='tight',dpi=250)
    plt.show()
# %%
def save_subgraph(name:str,subgraph:dict):
    with open(f'{name}.pkl','wb') as f:
        pickle.dump(subgraph,f)
# %%
# complete graph 6
subgraph_6 = csg.create_spanning_subgraph(nx.complete_graph(range(1,7)))
save_subgraph('subgraph_6',subgraph_6)
ans_6,eig_v_6= start(3,6,2,subgraph_6)
draw_result(ans_6,nx.complete_graph(range(1,7)),eig_v_6,3)
# %%
subgraph_7 = csg.create_spanning_subgraph(nx.complete_graph(range(1,8)))
save_subgraph('subgraph_7',subgraph_7)
ans_7,eig_v_7= start(3,7,2,subgraph_7)
draw_result(ans_7,nx.complete_graph(range(1,8)),eig_v_7,3)
# %%
subgraph_8 = csg.random_create_spanning_subgraph(nx.complete_graph(range(1,9)),500)
save_subgraph('subgraph_8',subgraph_8)
ans_8,eig_v_8 = start(3,8,2,subgraph_8)
draw_result(ans_8,nx.complete_graph(range(1,9)),eig_v_8,3)
# %%
subgraph_9 = csg.random_create_spanning_subgraph(nx.complete_graph(range(1,10)),1000)
save_subgraph('subgraph_9',subgraph_9)
ans_9,eig_v_9= start(3,9,2,subgraph_9)
draw_result(ans_9,nx.complete_graph(range(1,10)),eig_v_9,3)
# %%
for i in range(10,16):
    subgraph = csg.random_create_spanning_subgraph(nx.complete_graph(range(1,i+1)),1000)
    save_subgraph(f'subgraph_{i}',subgraph)
    ans,eig_v = start(3,i,2,subgraph)
    draw_result(ans,nx.complete_graph(range(1,i+1)),eig_v,3)
# %%
