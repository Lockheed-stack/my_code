# %%
import networkx as nx

def read_data():
    G = nx.read_edgelist('./hamster')
    return G
G = read_data()
nx.number_of_nodes(G)
# %%
def EnRenew(G,spreaders_num,local_infect_len):
    '''EnRenew,返回一组传播者
    ## Argument
    G: networkx Graph, type: nx.Graph()
    spreaders_num: 传播者的数量, type: int
    local_infect_len: 顶点的局部影响范围, type: int
    返回值: Result_set, type: list
    '''
    Result_set = []
    # 初始化,计算所有点的熵
    
    
    # 挑选传播者，根据顶点的局部影响范围，更新顶点的熵
    
    return Result_set