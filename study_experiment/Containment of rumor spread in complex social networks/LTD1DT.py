#%%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
#%%
class Graph_with_attr:
    def __init__(self,G:nx.Graph) -> None:
        self.G = nx.Graph(G)
        self.final_R_receiver = {}
        self.final_T_receiver = {}
    
    def copy(self,):
        G = nx.Graph(self.G)
        return self.__class__(G)

    def get_final_R_receiver(self):
        return self.final_R_receiver

    def get_final_T_receiver(self):
        return self.final_T_receiver
#%%
class LTD1DT_model:
    def __init__(self,GWA:Graph_with_attr,seed_R:list,seed_T:list,keep_original_data:bool=False) -> None:
        # self.final_R_receiver = {}
        # self.final_T_receiver = {}
        self.seed_T = seed_T.copy()
        self.seed_R = seed_R.copy()
        
        #self.G = nx.Graph(G)
        
        # init Graph's threshold & status
        if not keep_original_data:
            for node in nx.nodes(GWA.G):
                # self.G.nodes[node]['i_threshold']=np.random.uniform()
                # self.G.nodes[node]['d_threshold']=np.random.uniform()
                # self.G.nodes[node]['status']='inactive'
                GWA.G.nodes[node]['i_threshold']=np.random.uniform()
                GWA.G.nodes[node]['d_threshold']=np.random.uniform()
                GWA.G.nodes[node]['status']='inactive'
            # init R-seed nodes
            if len(seed_R) !=0:
                for node in seed_R:
                    GWA.G.nodes[node]['status']='R-active'
                    GWA.final_R_receiver[node]='R'
            # init T-seed nodes
            if len(seed_T) != 0:
                for node in seed_T:
                    GWA.G.nodes[node]['status']='T-active'
                    GWA.final_T_receiver[node]='T'

        #self.init_G = nx.Graph(self.G)
        self.GWA = GWA.copy()

    def copy(self):
        #G = nx.Graph(self.G)
        G = self.GWA.copy()
        return self.__class__(G,self.seed_R,self.seed_T)

    def diffusion_simulation(self,specified_G:Graph_with_attr=None):
        nothing_update = False
        time_steps = 0
        influenced_nodes = {}

        if specified_G is not None:
            G = specified_G.G
        else:
            G = self.GWA.G

        while(not nothing_update):

            nothing_update = True

            # influence stage
            for node in G.nodes():
                if G.nodes[node]['status']=='inactive':
                    if self.check_and_change_i_threshold(node,G):
                        nothing_update = False
                        influenced_nodes[node]=1
            
            # decision stage
            search_range = list(influenced_nodes.keys())+list(specified_G.get_final_R_receiver().keys())

            for node in search_range:
                if node  not in self.seed_R:
                    if self.check_and_change_d_threshold(node,influenced_nodes,specified_G):
                        nothing_update = False

            
            time_steps+=1



    def check_and_change_i_threshold(self,node,specified_G:nx.Graph=None):

        influenced_num = 0      
        node_deg = nx.degree(specified_G,node)

        for nbr in nx.neighbors(specified_G,node):
            if (specified_G.nodes[nbr]['status'] == 'R-active') or (specified_G.nodes[nbr]['status'] == 'T-active') :
                influenced_num+=1
        
        if (influenced_num/node_deg) >= specified_G.nodes[node]['i_threshold']:
            specified_G.nodes[node]['status'] = 'influenced'
            return True
        
        # neither R nor T can active this node
        return False


    def check_and_change_d_threshold(self,node,influence_nodes:dict,specified_G:Graph_with_attr=None):
        active_num = 0
        R_active_num = 0
        T_active_num = 0

        if specified_G is not None:
            G = specified_G.G
        else:
            G = self.GWA.G
        
        if (G.nodes[node]['status'] == 'R-active') or (G.nodes[node]['status'] == 'T-active'):
            return False


        for nbr in nx.neighbors(G,node):
            if G.nodes[nbr]['status'] == 'R-active':
                active_num+=1
                R_active_num+=1
            elif G.nodes[nbr]['status'] == 'T-active':
                active_num+=1
                T_active_num+=1
        
        if active_num == 0:
            return False

        if (R_active_num/active_num) > G.nodes[node]['d_threshold']:
            specified_G.get_final_R_receiver()[node]='R'
            G.nodes[node]['status']='R-active'
            if node in influence_nodes.keys():
                influence_nodes.pop(node)
            return True

        elif T_active_num > 0 :
            specified_G.get_final_T_receiver()[node]='T'
            G.nodes[node]['status']='T-active'
            if node in self.final_R_receiver:
                specified_G.get_final_R_receiver().pop(node)
            if node in influence_nodes.keys():
                influence_nodes.pop(node)
            return True
        
        return False
    
    
    def get_diffused_G(self):
        return nx.Graph(self.G)
    def get_init_G(self):
        return nx.Graph(self.init_G)
# %%
# %%
# %%
