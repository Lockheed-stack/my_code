# %%
from ctypes import sizeof
import networkx as nx
import math
def read_data():
    G = nx.read_edgelist('./hamster')
    return G
G = read_data()
G = nx.Graph(G)
G.number_of_edges()
G.degree['99']
nx.degree(G,'99')
# for node in nx.nodes(G):
#     print(node)
# %%
def EnRenew(G,spreaders_num,local_infect_len):
    '''EnRenew,返回一组传播者
    ## Argument
    G: networkx Graph, type: nx.Graph()
    spreaders_num: 传播者的数量, type: int
    local_infect_len: 顶点的局部影响范围, type: int
    返回值: Result_set, type: list
    '''
    Result_set = set()
    # 初始化,计算所有点的熵,并挑选出初始的最大熵的点
    init_entropy = {}
    for node in nx.nodes(G):
        nbr_degree=[]
        for nbr in nx.neighbors(G,node):
            nbr_degree.append(nx.degree(G,nbr))
            
        nbr_deg_sum = sum(nbr_degree)
        nbr_entropy_sum = 0
        for deg in nbr_degree:
            nbr_entropy_sum +=(-(deg/nbr_deg_sum)*math.log(deg/nbr_deg_sum))
        
        init_entropy[node]= nbr_entropy_sum
    
    Result_set.add(max(init_entropy,key=init_entropy.get))
        
    
    # 挑选传播者，根据顶点的局部影响范围，更新顶点的熵
    while Result_set.count()<spreaders_num:
        pass
        
    return Result_set
# %%
# 初始化,计算所有点的熵

testG = nx.Graph()
testG.add_nodes_from([i for i in range(1,17)])
testG.add_edges_from([(1,2),(1,3),(1,4),(1,5),(4,6)])
testG.add_edges_from([(2,x) for x in range(7,9)])
testG.add_edges_from([(5,x) for x in range(9,12)])
testG.add_edges_from([(3,x) for x in range(12,17)])
testG.degree
for edge in testG.edges:
    print(edge)
# %%
aa = {'a':456,'b':789,'c':444,'d':555}
aa.get('a')
# %%
