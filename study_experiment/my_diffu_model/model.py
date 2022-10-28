#%%
import networkx as nx
import numpy as np
import pandas as pd
import queue
#%%
class model:
    def __init__(self,G:nx.Graph,au_T_rate:float=0.2,org_R_rate:float=0.1,is_init:bool=False) -> None:
        '''Creating my model, improvement based on LT model

        Parameters
        ----------
        G : nx.Graph
            A Networkx's Graph.
        au_T_rate : float
            The percentage of authoritative Truth node.
        org_R_rate : float
            The percentage of organized Rumor node.
        is_init : bool
            False by default. If true, the model will use the original attributions of G.
        '''

        self.G = G.copy()
        
        # thest nodes will not change their status
        self.authoritative_T,self.organized_R = self._select_T_R_node(G,au_T_rate,org_R_rate)
        
        '''
        status:
            inactive:0
            R-active:1
            T-active:2
        '''
        if not is_init:
            for node in nx.nodes(G):
                self.G.nodes[node]['i_threshold'] = np.random.uniform() # influenced threshold
                self.G.nodes[node]['c_threshold'] = np.random.uniform() # correction threshold
                self.G.nodes[node]['status'] = 0
            
            # update new correction threshold
            for node in self.authoritative_T:
                self._new_correction_threshold(node,)

    
    def _new_correction_threshold(self,au_node,beta:float=1,n_th_nbr:int=4):
        '''Updating correction thresholds for nodes which under the influence by au_T nodes.

        Parameters
        ----------
        au_node : inferr
            The au_T node.
        beta : float, optional
            The power of au_T node's influence, by default 1. The smaller beta is, the more
            powerful ability of correction of au_T node.
        n_th_nbr : int, optional
            The range of au_T nodes'influence, by default 4.
        '''
        nbr_dict = {}
        nbr_dict[0] = [au_node]
        # nbr_dict[1] = list(nx.neighbors(self.G,au_node))

        # # update 1st neighbors' correction threshold
        # for node in nbr_dict[1]:
        #     p = self.G.nodes[node]['c_threshold']
        #     self.G.nodes[node]['c_threshold'] = p-p/(1+np.exp(beta))
        
        # 改成字典查询吧，列表有点慢。
        for i in range(0,n_th_nbr):
            nbr_dict[i+1] = [] # next level
            for node in nbr_dict[i]:
                if node in self.organized_R:
                    continue

                # next level nodes
                nbr_list = list(nx.neighbors(self.G,node))
                
                j=0
                while j < len(nbr_list):
 
                    if nbr_list[j] in self.authoritative_T:
                        nbr_list.pop(j)
                    # check current and next level
                    elif (nbr_list[j] in nbr_dict[i+1]) or (nbr_list[j] in nbr_dict[i]):
                        nbr_list.pop(j)

                    else:
                        p = self.G.nodes[nbr_list[j]]['c_threshold']
                        self.G.nodes[nbr_list[j]]['c_threshold'] = p-p/(1+np.exp(beta*(i+1)))
                        j+=1

                nbr_dict[i+1] += nbr_list
        

    def _select_T_R_node(self,G:nx.Graph,T_percent:float,R_percent:float):

        df = pd.DataFrame(G.degree,columns=['node','degree'],dtype=int).set_index('node')
        median_degree = np.median(list(dict(G.degree).values()))

        ge_median_node = df.query(f'degree > {median_degree}')

        T_sample_df = ge_median_node.sample(frac=T_percent,replace=False)
        T_nodes = list(T_sample_df.index)
        
        R_sample_df = ge_median_node.drop(T_sample_df.index,)
        R_nodes = list(R_sample_df.sample(frac=R_percent,replace=False).index)

        return T_nodes,R_nodes
#%%
# ------------- test -------------------
if __name__ == '__main__':
    model1 = model(nx.karate_club_graph(),)
#%%
G = nx.karate_club_graph()
# %%
dict(nx.neighbors(G,5),)
# %%
[0]*(10**8)
# %%
[0 for i in range(10**8)]
# %%
