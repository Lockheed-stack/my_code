# %%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
# %%
G =nx.read_adjlist('./SCHOLAT Social Network/links.txt',nodetype=int)
# %%
G.edges
# %%
nx.number_of_edges(G)
# %%
nx.number_of_nodes(G)
# %%
nx.number_of_cliques(G,12)
# %%
nx.number_connected_components(G)
# %%
nx.core_number(G,)[7186]
# %%
nx.degree(G,342)
# %%
for nbr in nx.neighbors(G,7186):
    print(nbr,G.degree[nbr])
# %%
G.degree(316,weight='whatIsThis')
# %%
new_g = nx.DiGraph(version='0.2')
new_g.add_weighted_edges_from([(1, 2, 0.75), (2, 3, 0.5)])
dict(new_g.degree(weight='weight'))
# %%
new_g.add_edge(1,3,lala=66)
dict(new_g.degree(weight='weight'))
# %%
G2 = nx.read_adjlist('./SCHOLAT Link Prediction/train.csv',delimiter=',',nodetype=int)

# %%
s2 = [G2.subgraph(c) for c in nx.connected_components(G2)]
# %%
s2_dia_dist = [nx.diameter(s2[i]) for i in range(1,164)]
# %%
s2_dia_dist
# %%
s2[0].number_of_nodes()
# %%
fig,ax = plt.subplots()
result = pd.value_counts(s2_dia_dist)
ax.set_title('SCHOLAT Link Prediction/train.csv')
ax.set_xlabel('component\'s diameter')
ax.set_ylabel('count')
ax.bar_label(ax.bar(result.index,result.values,),)
plt.tight_layout()
plt.savefig('./outpng/S_L_P_train',dpi=200,bbox_inches='tight')
plt.show()
# %%
result.values
# %%
