#%%
from random import seed
import networkx as nx
import numpy as np
import LTD1DT
# %%
class unconstrained_algorithm:
    def __init__(self,) -> None:
        # self.G = nx.Graph(G)
        #self.init_model  = diffusion_model.copy()
        pass
    
    
    
    def MinGreedy(self,model:LTD1DT.model_V2,k:int=1,seed_R:list=None):

        seed_T = []
        if seed_R is None:
            print('seed R has not  given.' )
            return seed_T
        if k <= 0:
            print('invaild k.')
            return seed_T
        
        test_model = model.copy()
        test_model.update_seed_R(seed_R,True)

        G , final_R , final_T = test_model.diffusion()
        final_R_num = len(final_R)
        

        while(len(seed_T)<k):
            
            chosen_node = None
            temp_model = test_model.copy()

            for node in nx.nodes(G):
                if (node not in seed_T) and (node not in seed_R):
                    temp_model.update_seed_T(seed_T+[node],True)
                    result = temp_model.diffusion()
                    if final_R_num > len(result[1]):
                        chosen_node = node
                        final_R_num = len(result[1])
                
                temp_model.update_seed_T(seed_T,True)
                
            if chosen_node is None:
                print(f'Cannot find {k} T nodes, only {len(seed_T)} found.')
                return seed_T
            else:
                seed_T.append(chosen_node)
                # temp_model.update_seed_T(seed_T)

        return seed_T
# %%

'''------------------ test below -----------------------'''
G = nx.Graph()
G.add_edges_from([(1,2),(1,7),(2,3),(3,4),(3,5),(4,6),(7,6),(7,8),(8,9),(9,5)])
nx.draw_networkx(G,pos=nx.circular_layout(G),with_labels=True,font_color='white')
#%%

# %%
