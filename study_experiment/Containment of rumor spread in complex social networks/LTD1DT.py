#%%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
#%%
class Graph_with_attr:
    def __init__(self,G:nx.Graph,final_R_receiver:dict=None,final_T_receiver:dict=None,) -> None:
        self.G = nx.Graph(G)

        if final_R_receiver is not None:
            self.final_R_receiver = final_R_receiver.copy()
        else:
            self.final_R_receiver = {}

        if final_T_receiver is not None:
            self.final_T_receiver = final_T_receiver.copy()
        else:
            self.final_T_receiver = {}
    
    def copy(self,):
        G = nx.Graph(self.G)
        return self.__class__(G,self.final_R_receiver,self.final_T_receiver)

    def get_final_R_receiver(self):
        return self.final_R_receiver.copy()

    def get_final_T_receiver(self):
        return self.final_T_receiver.copy()
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

    def diffusion_simulation(self,specified_G:Graph_with_attr=None,copy:bool=False):
        nothing_update = False
        time_steps = 0
        influenced_nodes = {}
        
        
        #if copy is False:

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
            if specified_G is not None:
                search_range = list(influenced_nodes.keys())+list(specified_G.get_final_R_receiver().keys())
            else:
                search_range = list(influenced_nodes.keys())+list(self.GWA.get_final_R_receiver().keys())

            for node in search_range:
                if node  not in self.seed_R:
                    if self.check_and_change_d_threshold(node,influenced_nodes,specified_G):
                        nothing_update = False

            
            time_steps+=1



    def check_and_change_i_threshold(self,node,specified_G:nx.Graph=None):

        influenced_num = 0      
        node_deg = nx.degree(specified_G,node)
        if specified_G is not None:
            G = specified_G
        else:
            G = self.GWA.G

        for nbr in nx.neighbors(G,node):
            if (G.nodes[nbr]['status'] == 'R-active') or (G.nodes[nbr]['status'] == 'T-active') :
                influenced_num+=1
        
        if (influenced_num/node_deg) >= G.nodes[node]['i_threshold']:
            G.nodes[node]['status'] = 'influenced'
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
            specified_G = self.GWA
        
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
            specified_G.final_R_receiver[node]='R'
            G.nodes[node]['status']='R-active'
            if node in influence_nodes.keys():
                influence_nodes.pop(node)
            return True

        elif T_active_num > 0 :
            specified_G.final_T_receiver[node]='T'
            G.nodes[node]['status']='T-active'
            if node in specified_G.final_R_receiver:
                specified_G.final_R_receiver.pop(node)
            if node in influence_nodes.keys():
                influence_nodes.pop(node)
            return True
        
        return False
    
    
    def update_seed_T(self,new_seed_T:list,cover:bool=True,specified_G:Graph_with_attr=None):
        '''Update specifiy GWA(Graph with attribution)'s seed_T
        
        Parameters
        -----------
        new_seed_T: list
            list of nodes

        cover: bool 
            Default True. If cover is True, then new seed_T will 
            cover the specified G's seed_T. Else new seed_T will append to original seed_T

        specified_G: Graph_with_attr
            Default None. If a GWA was specified, then will change the GWA's attrbution.
            Else the attribution of model's GWA will be changed.

        '''
        if cover:
            self.seed_T = new_seed_T.copy()
        else:
            self.seed_T+=new_seed_T

        if specified_G is not None:
            G = specified_G.G
        else:
            G = self.GWA.G
            specified_G = self.GWA

        for node in self.seed_T:
            G.nodes[node]['status']='T-active'
            specified_G.final_T_receiver[node]='T'
# %%
class model_V2:
    def __init__(self,G:nx.Graph,is_init:bool=False,seed_T:list=None,seed_R:list=None) -> None:
        '''Creating a LTD1TD model that running on G

        Parameters
        -----------
        G : nx.Graph
            A networkX's graph.
        is_init : bool
            Default False. If True, the model will use the original attributions of G.
        seed_T : list
            A list of T nodes.
        seed_R : list
            A list of R nodes.

        Note
        --------------
        If is_init is True and seedT or seedR is given(not None), the G's original attribuitions
        may be overrided.
        '''
        
        self.G = nx.Graph(G)
        self.final_R_receiver = {}
        self.final_T_receiver = {}

        if seed_T is not None:
            self.seed_T = seed_T.copy()
        else:
            self.seed_T = []
        if seed_R is not None:
            self.seed_R = seed_R.copy()
        else:
            self.seed_R = []

        # init Graph's threshold & status
        if not is_init:
            for node in nx.nodes(G):
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
        else:
            for node in nx.nodes(G):
                if G.nodes[node]['status'] == 'R-active':
                    self.final_R_receiver[node]='R'
                elif G.nodes[node]['status'] == 'T-active':
                    self.final_T_receiver[node] = 'T'
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

    
    def diffusion(self,is_apply:bool=False,seed_T:list=None,seed_R:list=None):
        '''Simulation test of diffusion

        Parameters
        ---------------
        is_apply : bool
            Default False. If True, then this simulation will be applied on model's G.

        Return
        --------------
        (G , final_R_receiver , final_T_receiver) : tuple
            A Networkx's Graph which simulated diffusion has been completed, 
            final_R_receiver and final_T_receiver
        '''
        if not is_apply:
            G = nx.Graph(self.G)
            final_T_receiver = self.final_T_receiver.copy()
            final_R_receiver = self.final_R_receiver.copy()
            
        else:
            G = self.G
            final_R_receiver = self.final_R_receiver
            final_T_receiver = self.final_T_receiver

        # if seed_T is not None:
        #     for node in seed_T:
        #         G.nodes[node]['status']='T-active'
        #         final_T_receiver[node] = 'T'
        
        # nothing_change = False
        R_num = -1
        T_num = -1
        while(R_num-len(final_R_receiver) or T_num-len(final_T_receiver)):

            # nothing_change = True
            R_num = len(final_R_receiver)
            T_num = len(final_T_receiver)
            # influence stage
            for node in nx.nodes(G):
                if G.nodes[node]['status'] == 'inactive':
                    if self.__check_i_threshold(node,G):
                        # nothing_change = False
                        pass

            # decision stage
            for node in nx.nodes(G):
                if (G.nodes[node]['status'] == 'influenced') or (G.nodes[node]['status'] == 'R-active'):
                    if node not in self.seed_R:
                        flag = self.__check_d_threshold(node,G)
                        if flag != 0:
                            # nothing_change = False
                            if flag == 1:
                                final_R_receiver[node]='R'
                                G.nodes[node]['status'] = 'R-active'
                            elif flag == 2:
                                final_T_receiver[node]='T'
                                G.nodes[node]['status']='T-active'
                                if node in  final_R_receiver:
                                    final_R_receiver.pop(node)
        
        return G , final_R_receiver , final_T_receiver

    def __check_i_threshold(self,node,G:nx.Graph):
        influenced_num = 0
        node_deg = nx.degree(G,node)
        if node_deg == 0:
            return False

        for nbr in nx.neighbors(G,node):
            if (G.nodes[nbr]['status'] == 'R-active') or (G.nodes[nbr]['status'] == 'T-active') :
                influenced_num+=1
        
        if (influenced_num/node_deg) >= G.nodes[node]['i_threshold']:
            G.nodes[node]['status'] = 'influenced'
            return True
        
        return False
    
    def __check_d_threshold(self,node,G:nx.Graph,):
        active_num = 0
        R_active_num = 0
        T_active_num = 0

        if (G.nodes[node]['status'] == 'T-active'):
            return 0
        
        for nbr in nx.neighbors(G,node):
            if G.nodes[nbr]['status'] == 'R-active':
                active_num+=1
                R_active_num+=1
            elif G.nodes[nbr]['status'] == 'T-active':
                active_num+=1
                T_active_num+=1
        
        if active_num == 0:
            return 0

        if (R_active_num/active_num) > G.nodes[node]['d_threshold']:
            return 1

        elif T_active_num > 0 :
            return 2
        
        return 0

    def update_seed_T(self,seed_T:list,override:bool=False):
        '''Updating the model's attribution associated with seed_T(include seed_T).

        Parameters
        ----------------
        seed_T : list
            a list of T nodes.
        override : bool
            Default False. If True, the new seed_T will cover original seed_T and remove
            the relative attributions of model's G to apply the new seed_T. 
            Else the new seed_T will append to original seed_T.

        Note
        -------------
        Only inactive node will be changed. If status of node is R-active or T-active, it will
        omit this update.
        '''
        
        
        if (seed_T is None):
            return 

        if not override:
            self.seed_T += seed_T

            for node in seed_T:
                if self.G.nodes[node]['status'] == 'inactive':
                    self.G.nodes[node]['status'] = 'T-active'
                    self.final_T_receiver[node]='T'
        else:
            for node in self.seed_T:
                self.G.nodes[node]['status'] = 'inactive'
                self.final_T_receiver.pop(node)
            for node in seed_T:
                self.G.nodes[node]['status'] = 'T-active'
                self.final_T_receiver[node]='T'
            self.seed_T = seed_T.copy()

    def update_seed_R(self,seed_R:list,override:bool=False):
        '''Updating the model's attribution associated with seed_R(include seed_R).

        Parameters
        ----------------
        seed_R : list
            a list of T nodes.
        override : bool
            Default False. If True, the new seed_R will cover original seed_R and remove
            the relative attributions of model's G to apply the new seed_R. 
            Else the new seed_R will append to original seed_R.

        Note
        -------------
        Only inactive node will be changed. If status of node is R-active or T-active, it will
        omit this update.
        '''
        
        
        if (seed_R is None):
            return 

        if not override:
            self.seed_R += seed_R

            for node in seed_R:
                if self.G.nodes[node]['status'] == 'inactive':
                    self.G.nodes[node]['status'] = 'R-active'
                    self.final_R_receiver[node]='R'
        else:
            for node in self.seed_R:
                self.G.nodes[node]['status'] = 'inactive'
                self.final_R_receiver.pop(node)
            for node in seed_R:
                self.G.nodes[node]['status'] = 'R-active'
                self.final_R_receiver[node]='R'
            self.seed_R = seed_R.copy()



    def get_initialized_G(self,):
        return nx.Graph(self.G)

    def copy(self,):
        G = nx.Graph(self.G)
        return self.__class__(G,True,self.seed_T.copy(),self.seed_R.copy())
# %%

# %%
