#%%
import networkx as nx
import numpy as np
import LTD1DT
# %%
class unconstrained_algorithm:
    def __init__(self,) -> None:
        # self.G = nx.Graph(G)
        #self.init_model  = diffusion_model.copy()
        pass

    def MinGreedy(self,model:LTD1DT.LTD1DT_model,GWA:LTD1DT.Graph_with_attr,k:int,seed_R:list):
        seed_T = []
        initialized_GWA = GWA.copy()

        temp_model = model.copy()
        temp_model.diffusion_simulation(initialized_GWA,)
        final_R_num = len(initialized_GWA.final_R_receiver)
        
        
        while(len(seed_T)<k):
            choosen_node = None
            temp_GWA = initialized_GWA.copy()

            # find nodes which minimize the spread of rumor
            for node in temp_GWA.G.nodes():
                
                if (node not in seed_R) and (node not in seed_T):
                    model.update_seed_T([node],temp_GWA)
                    model.diffusion_simulation(temp_GWA) 
                    if final_R_num > len(temp_GWA.get_final_R_receiver()):
                        final_R_num = len(temp_GWA.get_final_R_receiver())
                        choosen_node = node
                    else:
                        temp_GWA = initialized_GWA.copy()
            
            if choosen_node is not None:
                seed_T.append(choosen_node)
                model.update_seed_T(seed_T,initialized_GWA)
            else:
                print(f'can not find k nodes of T_seed, only {len(seed_T)} nodes found.')
                return seed_T
        
        return seed_T
# %%

'''------------------ test below -----------------------'''
G = nx.Graph()
G.add_edges_from([(1,2),(1,7),(2,3),(3,4),(3,5),(4,6),(7,6),(7,8),(8,9),(9,5)])
nx.draw_networkx(G,pos=nx.circular_layout(G),with_labels=True,font_color='white')
#%%
GWA = LTD1DT.Graph_with_attr(G)
model = LTD1DT.LTD1DT_model(GWA,[],[])
initialized_GWA = model.GWA.copy()
# %%
nx.get_node_attributes(model.GWA.G,'i_threshold')
# %%
nx.get_node_attributes(initialized_GWA.G,'status')
# %%
# %%
un_al = unconstrained_algorithm()
un_al.MinGreedy(model,initialized_GWA,3,[6,8,])
# %%
nx.get_node_attributes(initialized_GWA.G,'status')
# %%
