#%%
import networkx as nx
import numpy as np
import LTD1DT
# %%
class unconstrained_algorithm:
    def __init__(self,G:nx.Graph(),) -> None:
        self.G = G
        
    def MinGreedy(self,k:int,seed_R:list):
        seed_T = []
        
        while(len(seed_T)<k):
            model_1 = LTD1DT.LTD1DT_model(self.G,seed_R,seed_T)
            model_1.diffusion_simulation()
            final_R_num = len(model_1.get_final_R_receiver())
            choosen_node = None

            # find nodes which minimize the spread of rumor
            for node in self.G.nodes():
                if (node not in seed_R) and (node not in seed_T):
                    model_2 = LTD1DT.LTD1DT_model(self.G,seed_R,seed_T+[node])
                    model_2.diffusion_simulation()
                    if final_R_num > len(model_2.get_final_R_receiver()):
                        final_R_num = len(model_2.get_final_R_receiver())
                        choosen_node = node
            seed_T.append(choosen_node)
# %%
