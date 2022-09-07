#%%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
#%%
class LTD1DT_model:
    def __init__(self,G:nx.Graph,seed_R:list=None,seed_T:list=None) -> None:
        self.final_R_receiver = {}
        self.final_T_receiver = {}
        self.seed_T = seed_T.copy()
        self.seed_R = seed_R.copy()

        self.G = nx.Graph(G)
        
        # init Graph's threshold & status
        for node in nx.nodes(self.G):
            self.G.nodes[node]['i_threshold']=np.random.uniform()
            self.G.nodes[node]['d_threshold']=np.random.uniform()
            self.G.nodes[node]['status']='inactive'
        # init R-seed nodes
        if len(seed_R) !=0:
            for node in seed_R:
                self.G.nodes[node]['status']='R-active'
                self.final_R_receiver[node]='R'
        # init T-seed nodes
        if len(seed_T) != 0:
            for node in seed_T:
                self.G.nodes[node]['status']='T-active'
                self.final_T_receiver[node]='T'

        self.init_G = nx.Graph(self.G)

    def copy(self):
        G = nx.Graph(self.G)

        return self.__class__(G,self.seed_R,self.seed_T)

    def diffusion_simulation(self,):
        nothing_update = False
        time_steps = 0
        influenced_nodes = {}

        while(not nothing_update):

            # RT_list = list(self.final_R_receiver.keys())+list(self.final_T_receiver.keys())

            # R & T spread process(influence stage)
            
            for node in self.G.nodes():
                if self.G.nodes[node]['status']=='inactive':
                    if self.check_and_change_i_threshold(node,):
                        nothing_update = False
                        influenced_nodes[node]=1
                else:
                    nothing_update = True
            # decision stage
            search_range = list(influenced_nodes.keys())+list(self.final_R_receiver.keys())

            for node in search_range:
                if node  not in self.seed_R:
                    if self.check_and_change_d_threshold(node,):
                        nothing_update = False
                else:
                    nothing_update = True
            
            time_steps+=1



    def check_and_change_i_threshold(self,node,):
        node_deg = nx.degree(self.G,node)
        # R_curr_value = 0
        # T_curr_value = 0
        influenced_num = 0

        for nbr in nx.neighbors(self.G,node):
            if (self.G.nodes[nbr]['status'] == 'R-active') or (self.G.nodes[nbr]['status'] == 'T-active') :
                influenced_num+=1
        
        if (influenced_num/node_deg) >= self.G.nodes[node]['i_threshold']:
            self.G.nodes[node]['status'] = 'influenced'
            return True

        # if (R_curr_value/node_deg) >= self.G.nodes[node]['i_threshold']:
        #     self.G.nodes[node]['status']='R-active'
        #     self.final_R_receiver[node]='R'
        #     # return 1
        #     return True

        # elif (T_curr_value/node_deg) >= self.G.nodes[node]['i_threshold']:
        #     self.G.nodes[node]['status']='T-active'
        #     self.final_T_receiver[node]='T'
        #     #return 2
        #     return True
        
        # neither R nor T can active this node
        return False


    def check_and_change_d_threshold(self,node,):
        active_num = 0
        R_active_num = 0

        # if is_apply:
        #     G = nx.Graph(diffu_sim_G)
        #     final_R_receiver = R.copy()
        #     final_T_receiver = T.copy()
        # else:
        #     G = self.G
        #     final_R_receiver = self.final_R_receiver
        #     final_T_receiver = self.final_T_receiver

        for nbr in nx.neighbors(self.G,node):
            if self.G.nodes[nbr]['status'] == 'R-active':
                active_num+=1
                R_active_num+=1
            if self.G.nodes[nbr]['status'] == 'T-active':
                active_num+=1
        
        if active_num == 0:
            return False

        if (R_active_num/active_num) < self.G.nodes[node]['d_threshold']:
            self.final_T_receiver[node]='T'
            self.G.nodes[node]['status']='T-active'
            if node in self.final_R_receiver:
                self.final_R_receiver.pop(node)

            return True
        
        return False


    def update_T_seed_in_init_G(self,seed_T:list):
        
        self.seed_T+= seed_T
        # update T-seed nodes
        if len(seed_T) != 0:
            for node in seed_T:
                if self.G.nodes[node]['status'] != 'R-active':
                    self.G.nodes[node]['status']='T-active'
                    self.final_T_receiver[node]='T'


    def get_final_R_receiver(self):
        return self.final_R_receiver

    def get_final_T_receiver(self):
        return self.final_T_receiver
    
    def get_diffused_G(self):
        return nx.Graph(self.G)
    def get_init_G(self):
        return nx.Graph(self.init_G)
# %%
# %%
def a(b:list):
    b+[1]
aa = [1,23]
a(aa)
aa
# %%
