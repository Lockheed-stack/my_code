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
G.number_of_nodes() # (仅这段代码)共8个点，1,2,3,s,p,a,m,"spam"
G.number_of_edges() # 3条边

G.clear() # 清除所有点和边
# %%

