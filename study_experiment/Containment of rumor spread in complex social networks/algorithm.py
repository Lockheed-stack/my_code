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

    def MinGreedy(self,GWA:LTD1DT.Graph_with_attr,k:int,seed_R:list):
        seed_T = []
        model_1 = LTD1DT.LTD1DT_model(GWA,seed_R,seed_T)
        model_1.diffusion_simulation()# GWA.G didn't change
        final_R_num = len(model_1.GWA.final_R_receiver())
        
        
        while(len(seed_T)<k):
            choosen_node = None
            temp_GWA = GWA.copy()
            # find nodes which minimize the spread of rumor
            for node in temp_GWA.G.nodes():

                if (node not in seed_R) and (node not in seed_T):
                    model_1.diffusion_simulation(temp_GWA)
                    if final_R_num > len(temp_GWA.get_final_R_receiver()):
                        final_R_num = len(temp_GWA.get_final_R_receiver())
                        choosen_node = node
            
            if choosen_node is not None:
                seed_T.append(choosen_node)
                model_1.seed_T.append(choosen_node)
            else:
                print(f'can not find k nodes of T_seed, only {len(seed_T)} nodes found.')
                return seed_T
        
        return seed_T
# %%

'''------------------ test below -----------------------'''
G = nx.path_graph(range(1,15))
GWA = LTD1DT.Graph_with_attr(G)
model = LTD1DT.LTD1DT_model(GWA,[6,8],[])
# %%
nx.get_node_attributes(GWA.G,'status')
# %%
model.diffusion_simulation(GWA)
# %%

# %%
