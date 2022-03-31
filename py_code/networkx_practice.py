# %%
import networkx as nx
# %%
'''
顶点node的操作
'''
G = nx.Graph()
G.add_node(1)
G.add_node("hello")
G.add_nodes_from([4,6]) #参数为可迭代对象
G.add_nodes_from([
    (7,{"color":"red"}),
    (8,{"color":"blue"})
    ])
H = nx.path_graph(10) #从其他图获取顶点
G.add_nodes_from(H)
G.add_node(H) #或者直接把H当成顶点
# %%
'''
边edge的操作
'''
G.add_edge(1,2)
e = (2,3)
G.add_edge(*e) # unpack edge tuple
G.add_edges_from([(1,2),(3,4)]) # using a list of edge tuples
# An edge-tuple can be a 2-tuple of nodes 
# or a 3-tuple with 2 nodes followed 
# by an edge attribute dictionary
G.add_edges_from([
    (4,5,{"weight":3.14}),
    (5,6,{"weight":2.71})
])

# 添加已有的边/点的操作会被忽略
G.add_edges_from([(1, 2), (1, 3)])
G.add_node(1)
G.add_edge(1, 2)
G.add_node("spam")        # adds node "spam"
G.add_nodes_from("spam")  # adds 4 nodes: 's', 'p', 'a', 'm'
G.add_edge(3, 'm')
G.number_of_nodes() # (仅这段代码,34~42行)共8个点，1,2,3,s,p,a,m,"spam"
G.number_of_edges() # 3条边


# %%
DG = nx.DiGraph() 
DG.add_edge(2, 1)   # adds the nodes in order 2, 1 
DG.add_edge(1, 3) 
DG.add_edge(2, 4)
DG.add_edge(1, 2) 
# 顶点的邻接顺序是按照边添加的顺序排列的
assert list(DG.successors(2)) == [1, 4] 
# 然而，G.edges 的顺序是邻接的顺序，包括节点的顺序和每个节点的邻接。
assert list(DG.edges) == [(2, 1), (2, 4), (1, 3), (1, 2)]
# %%
'''
查看图中的元素
'''
list(G.nodes)
list(G.edges)
list(G.neighbors(2))
G.edges([2,'m']) # 查看与顶点2或者顶点m关联的边
G.degree([2,3]) # 查看顶点2和顶点3的度

# %%
'''删除图中元素'''
G.remove_node(2)
G.remove_nodes_from("spam")
G.remove_edge(1,3)
G.clear() # 清除所有点和边
# %%
'''不用一个个添加元素来生成图,直接“一次性”搞定'''
G.add_edge(1, 2) 
H = nx.DiGraph(G)  # create a DiGraph using the connections from G 
print(list(H.edges()))
edgelist = [(0, 1), (1, 2), (2, 3)] 
H = nx.Graph(edgelist)  # create a graph from an edge list 
print(list(H.edges()))
adjacency_dict = {0: (1, 2), 1: (0, 2), 2: (0, 1)} 
H = nx.Graph(adjacency_dict)  # create a Graph dict mapping nodes to nbrs 
list(H.edges()) 
# %%
'''访问图中的edge和neighbors'''
G = nx.Graph([(1,2,{"color":"yellow"})])
# 下标（数组）的形式访问
G[1] #访问顶点1的邻居，the same as G.adj[1]
G[1][2] #访问 (1,2)这条边
G.edges[1,2]

# 给一条边设置属性
G.add_edge(1, 3)
G[1][3]['color'] = "blue"
G.edges[1, 2]['color'] = "red"
G.edges[1, 2]

# %%
