#%%
import networkx as nx
import numpy as np
import LTD1DT
# %%
class unconstrained_algorithm:
    def __init__(self,G:nx.Graph,diffusion_model:LTD1DT.LTD1DT_model=None) -> None:
        self.G = nx.Graph(G)
        #self.init_model  = diffusion_model.copy()

    def MinGreedy(self,k:int,seed_R:list):
        seed_T = []
        init_G = nx.Graph(self.G)
        model_1 = LTD1DT.LTD1DT_model(init_G,seed_R,seed_T)
        model_1.diffusion_simulation(seed_T)
        final_R_num = len(model_1.get_final_R_receiver())
        
        
        while(len(seed_T)<k):
            choosen_node = None
            # find nodes which minimize the spread of rumor
            for node in self.G.nodes():
                init_G = nx.Graph(self.G)
                if (node not in seed_R) and (node not in seed_T):
                    model_2 = LTD1DT.LTD1DT_model(init_G,seed_R,seed_T+[node])
                    model_2.diffusion_simulation()
                    if final_R_num > len(model_2.get_final_R_receiver()):
                        final_R_num = len(model_2.get_final_R_receiver())
                        choosen_node = node
            
            if choosen_node is not None:
                seed_T.append(choosen_node)
            else:
                print(f'can not find k nodes of T_seed, only {len(seed_T)} nodes found.')
                return seed_T
        
        return seed_T
# %%

'''------------------ test below -----------------------'''
# %%
