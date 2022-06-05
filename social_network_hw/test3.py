# %%
import scipy.stats
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# %%
G = nx.read_adjlist('./SCHOLAT Link Prediction/train.csv',nodetype=int,delimiter=',')
# %%
def Adamic_Adar(G,node_x):
    # AA_index={}
    # for node_y in G.nodes():
    #     if node_y!=node_x:
    #         aa_value=0
    #         for node in nx.neighbors(G,node_y):
    #             if node in nx.neighbors(G,node_x):
    #                 aa_value+=1/np.log(nx.degree(G,node))
    #         AA_index[node_y]=aa_value
    # # node_y_nbrs = nx.neighbors(G,264)
    # # node_x_nbrs = nx.neighbors(G,node_x)
    # # for i in node_x_nbrs:
    # #     print(f"x_nbrs:{i}")
    # # for i in node_y_nbrs:
    # #     print(f"y_nbrs:{i}")
    # # aa_value = 0
    # # for node in node_y_nbrs:
    # #     if node in node_x_nbrs:
    # #         aa_value+=1.0/np.log(nx.degree(G,node))
    # # AA_index[264]=aa_value
    # return AA_index
    
    AA_index={}
    for node_y in G.nodes():
        if node_y!=node_x:
            aa_value=0
            for node in nx.neighbors(G,node_y):
                if node in nx.neighbors(G,node_x):
                    aa_value+=1/np.log(nx.degree(G,node))
            AA_index[node_y]=aa_value
            
    trimed_AA_index={}
    for node in AA_index.keys():
        if node not in nx.neighbors(G,node_x):
            
            trimed_AA_index[node]=AA_index[node]
            
    return trimed_AA_index
# %%
output=Adamic_Adar(G,126)
output
# %%
output[243]
# %%
sorted(output.items(),key=lambda kv:kv[1],reverse=True)
# %%
assert 264 not in nx.neighbors(G,126)
# %%
a=0
for node in nx.nodes(G):
   if node not in nx.neighbors(G,126):
       print(node)
# %%
b=0
for node_y in nx.neighbors(G,126):
    if node_y in nx.neighbors(G,264):
        print(node_y,1/np.log(nx.degree(G,node_y)))
        b+=1.0/np.log(nx.degree(G,node_y))
        
b
# %%
for u,v,p in nx.adamic_adar_index(G,[(126,243)]):
    print(f"({u},{v})-->{p}")
# %%
assert 268 in nx.neighbors(G,126)
# %%
for i in nx.common_neighbors(G,61,126):
    print(nx.degree(G,i))
# %%
G.number_of_nodes()
# %%
np.shape(output)
# %%
len(output)
# %%
s_g = [G.subgraph(c) for c in nx.connected_components(G)]
s_g[2]
# %%
len(s_g[10])
# %%
nx.draw_networkx(s_g[4],with_labels=True)
# %%
output = Adamic_Adar(s_g[4],1490)
output
# %%
assert 1080 in nx.neighbors(s_g[4],1490)
# %%
for u,v,p in nx.adamic_adar_index(s_g[4],[(1080,1490)]):
    print(f"({u},{v})-->{p}")
# %%
def test(G,node_x):
    AA={}
    for node_y in G.nodes():
        if node_y!=node_x:
            aa=0
            for node in nx.neighbors(G,node_y):
                if node in nx.neighbors(G,node_x):
                    
                    aa+=1/np.log(nx.degree(G,node))
            AA[node_y]=aa
    
    BB={}
    for node in AA.keys():
        if node not in nx.neighbors(G,node_x):
            if AA[node]!=0:
                BB[node]=AA[node]
    
    return BB
# %%
bb = test(G,6899)
# %%
sorted(bb.items(),key=lambda kv:kv[1],reverse=True)
# %%
bb
# %%
nx.degree(G,268)
# %%        
dg={}
for s in s_g:
    if 268 in nx.nodes(s):

        for node in nx.nodes(s):
            dg[node]=(nx.degree(s,node))
dg
# %%
sorted(dg.items(),key=lambda kv:kv[1],)
# %%
len(dg_stat)
# %%
len(s_g[0])
# %%
dg_stat={}
for node in G.nodes():
    if nx.degree(G,node) not in dg_stat:
        dg_stat[nx.degree(G,node)]=1
    else:
        dg_stat[nx.degree(G,node)]+=1
dg_stat
# %%
dg_stat=sorted(dg_stat.items(),key=lambda kv:kv[0],)
dg_stat
# %%
x=[]
y=[]
for i in dg_stat:
    x.append(i[0])
    y.append(i[1])
x
#%% 
fig,ax=plt.subplots(1,1)
plt.bar(x,y)
ax.set_xlabel('degree')
ax.set_ylabel('num')
plt.savefig('./test.jpg',bbox_inches='tight',dpi=300,)
plt.show()
# %%
a = [[1,3,4],[2,5,66],[24,43,23]]
# %%
a = np.array([0,0,0,1])
b = np.array([[0,1/3,1/3,1/3],
             [0.5,0,0,0.5],
             [0,0,0,1],
             [0,1,0,0]])
for i in  np.arange(100):
    a=np.dot(a,b)
    print(a)
# %%
(6/15)*np.log(6/15)
# %%
scipy.stats.entropy([1/3,2/3])
# %%
0.6931/(1.38629+0.6931)
# %%
type(b)
# %%
import Adamic_Adar as AA
outputAA = AA.Adamic_Adar(G,126)
sorted(outputAA.items(),key=lambda kv:kv[1],reverse=True)
# %%
