#%%
import networkx as nx
from numpy import random,exp
import pandas as pd
from queue import SimpleQueue
from random import randint
#%%
class model:

    def __init__(self, G: nx.Graph, au_T_rate: float = 0.01,) -> None:
        '''Creating my model, improvement based on LT model

        Parameters
        ----------
        G : nx.Graph
            A Networkx's Graph.
        au_T_rate : float
            The percentage of authoritative Truth node.
        '''

        self.G:nx.Graph = G.copy()
        self.droped_auT_df = pd.DataFrame()
        self.df = pd.DataFrame(self.G.degree, columns=['node', 'degree'], dtype=int).set_index('node')
        self.stubborn_R = {} # the seed of rumor nodes

        self.authoritative_T = self.select_authoritative_T_nodes(au_T_rate)
        self.other_T = {} # the monitoring T nodes or other T nodes
        '''
        group:
            inactive:0
            R-active:1
            T-active:2
        '''
        for node in nx.nodes(G):
            self.G.nodes[node]['i_threshold'] = random.uniform()  # influenced threshold
            self.G.nodes[node]['c_threshold'] = random.uniform()  # correction threshold
            self.G.nodes[node]['group'] = 0

        # update new correction threshold
        for node in self.authoritative_T.keys():
            self.G.nodes[node]['group'] = 2
            self.__new_correction_threshold(node,)

    def __new_correction_threshold(self, au_node, beta: float = 1, n_th_nbr: int = 3):
        '''Updating correction thresholds for nodes which under the influence by au_T nodes.

        Parameters
        ----------
        au_node : inferr
            The au_T node.
        beta : float, optional
            The power of au_T node's influence, by default 1. The smaller beta is, the more
            powerful ability of correction of au_T node.
        n_th_nbr : int, optional
            The range of au_T nodes'influence, by default 3.
        '''
        nbr_queue = SimpleQueue()
        checked_node = {au_node: 1}

        for nbr in nx.neighbors(self.G, au_node):
            nbr_queue.put((nbr, 1))  # (node,k-th nbr)
            checked_node[nbr] = 1

        while not nbr_queue.empty():
            node, order = nbr_queue.get()

            # update new correction threshold
            p = self.G.nodes[node]['c_threshold']
            self.G.nodes[node]['c_threshold'] = p - p / (1 + exp(beta * (order)))

            if order < n_th_nbr:
                for nbr in nx.neighbors(self.G, node):
                    if nbr in checked_node:
                        continue
                    nbr_queue.put((nbr, order + 1))
                    checked_node[nbr] = 1


    def select_authoritative_T_nodes(self, T_percent: float):

        #df = pd.DataFrame(self.G.degree, columns=['node', 'degree'], dtype=int).set_index('node')
        if T_percent == 0:
            self.droped_auT_df = self.df.copy()
            return {}
            
        median_degree = self.df['degree'].median()

        ge_median_nodes = self.df.query(f'degree>{median_degree}')

        if (ge_median_nodes.shape[0]*T_percent) < 1:
            T_sample_df = ge_median_nodes.sample(1, replace=False)
        else:
            T_sample_df = ge_median_nodes.sample(frac=T_percent, replace=False)

        T_nodes = T_sample_df.to_dict('index')
        
        self.droped_auT_df = self.df.drop(T_sample_df.index,)
        return T_nodes

    def __check_i_threshold(self, node, G: nx.Graph):
        '''check influence threshold

        Parameters
        ----------
        node : nx.Graph node
            a nx.Graph node
        G : nx.Graph
            a networkx

        Returns
        -------
        int
            0 : nothing change
            1 : activated by rumor
            2 : activated by truth
        '''
        #influence_num = 0
        T_num, R_num = 0, 0
        node_deg = nx.degree(G, node)
        if node_deg == 0:
            return 0

        for nbr in nx.neighbors(G, node):
            if G.nodes[nbr]['group'] != 0:
                #influence_num +=1
                if G.nodes[nbr]['group'] == 1:
                    R_num += 1
                else:
                    T_num += 1

        # Priority to become a rumor node
        if (R_num / node_deg) >= G.nodes[node]['i_threshold']:
            return 1
        elif (T_num / node_deg) >= G.nodes[node]['i_threshold']:
            return 2

        return 0

    def __check_c_threshold(self,node,G:nx.Graph):
        actived_num = 0
        T_active_num = 0

        for nbr in nx.neighbors(G,node):
            check_status = G.nodes[nbr]['group']

            if check_status != 0:
                actived_num +=1
            if check_status == 2:
                T_active_num +=1
        
        if actived_num == 0:
            return False
        elif (T_active_num/actived_num) >= G.nodes[node]['c_threshold']:
            return True

        return False     

    def generate_R_nodes(self, org_R_rate:float=0.0, all_R_rate:float=0.05, other_T_node:list=None):
        
        if org_R_rate > all_R_rate:
            return []
        if other_T_node is not None:
            droped_T_df = self.droped_auT_df.drop(other_T_node)
        else:
            droped_T_df = self.droped_auT_df

        # organized R nodes and normal R nodes
        if org_R_rate>0:
            median_degree = droped_T_df['degree'].median()
            ge_median_df = droped_T_df.query(f'degree>={median_degree}')

            if (org_R_rate*ge_median_df.shape[0]) <1:
                org_R_nodes = list(ge_median_df.sample(1).index)
            else:
                org_R_nodes = list(ge_median_df.sample(frac=org_R_rate,replace=False).index)
                
            droped_T_orgR_df = droped_T_df.drop(org_R_nodes)
            if ((all_R_rate-org_R_rate)*droped_T_orgR_df.shape[0])<1:
                normal_R_nodes = list(droped_T_orgR_df.sample(1))
            else:
                normal_R_nodes = list(droped_T_orgR_df.sample(frac=(all_R_rate-org_R_rate),replace=False).index)
            
            self.stubborn_R.clear()
            for node in org_R_nodes + normal_R_nodes:
                self.stubborn_R[node]=1

            return org_R_nodes + normal_R_nodes
        
        # only normal R nodes
        else:
            if (all_R_rate*droped_T_df.shape[0])<1:
                normal_R_nodes = list(droped_T_df.sample(n=1,replace=False).index)
            else:
                normal_R_nodes = list(droped_T_df.sample(frac=all_R_rate,replace=False).index)
            
            self.stubborn_R.clear()
            for node in normal_R_nodes:
                self.stubborn_R[node]=1

            return normal_R_nodes

    def refresh_i_c_threshold(self,new_au_T:bool=False,au_T_rate:float=0.0):
        
        if new_au_T:
            self.authoritative_T.clear()
            self.authoritative_T = self.select_authoritative_T_nodes(au_T_rate)
        
        for node in nx.nodes(self.G):
            self.G.nodes[node]['i_threshold'] = random.uniform()  # influenced threshold
            self.G.nodes[node]['c_threshold'] = random.uniform()  # correction threshold
            self.G.nodes[node]['group'] = 0
        
        for node in self.authoritative_T.keys():
            self.G.nodes[node]['group'] = 2
            self.__new_correction_threshold(node,)

    def update_normal_T_nodes(self,G:nx.Graph,seed_T:list,override:bool=True):
        if override:
            for node in self.other_T.keys():
                G.nodes[node]['group'] = 0
            
            self.other_T.clear()
            for node in seed_T:
                G.nodes[node]['group'] = 2
                self.other_T[node]=1
        else:
            for node in seed_T:
                G.nodes[node]['group'] = 2
                self.other_T[node] = 1

    def update_auT_nodes(self,G:nx.Graph,T_percent:float,override:bool=True):
        if override:
            pass

    def unblocking_diffusion(self,seed_R_nodes:list,is_apply:bool=False):
        if not is_apply:
            G:nx.Graph = self.G.copy()
        else:
            G = self.G
        
        spread_time = 0
        final_R_receiver = {}
        R_t_receiver_num = {0: len(seed_R_nodes)}
        search_range = SimpleQueue()

        for node in self.authoritative_T.keys():
            G.nodes[node]['group'] = 0

        # init final_R_receiver & search_range
        for node in seed_R_nodes:
            final_R_receiver[node] = 1
            G.nodes[node]['group'] = 1
            G.nodes[node]['active_time'] = spread_time # record the active time of rumor node
            for nbr in nx.neighbors(G, node):
                if G.nodes[nbr]['group'] == 0:
                    search_range.put(nbr)

        nothing_change = False

        while (not nothing_change):

            nothing_change = True
            circulation_times = search_range.qsize()
            spread_time += 1

            for i in range(circulation_times):
                node = search_range.get()
                if G.nodes[node]['group'] == 1: # This node is R-active (the search queue has two same nodes, the first node has been actived by rumor)
                    continue
                if self.__check_i_threshold(node, G) == 1:  # actived by rumor
                    nothing_change = False
                    G.nodes[node]['group'] = 1
                    G.nodes[node]['active_time'] = spread_time
                    final_R_receiver[node] = 1

                    for nbr in nx.neighbors(G, node):
                        if G.nodes[nbr]['group'] == 0:
                            search_range.put(nbr)
            if not nothing_change:
                R_t_receiver_num[spread_time] = len(final_R_receiver)

        return G, spread_time, final_R_receiver, R_t_receiver_num

    def before_detected_diffusion(self, seed_R_nodes: list, T_nodes: list=None, is_apply: bool = False):
        '''Simulation test of diffusion

        Parameters
        ----------
        seed_R_nodes : list
            the rumor seed nodes
        T_nodes : list
            the monitoring T node
        is_apply : bool, optional
            If True, then this simulation will be applied on model's G, by default False
     
        '''
        if not is_apply:
            G:nx.Graph = self.G.copy()
        else:
            G = self.G
        if T_nodes is None:
            T_nodes = []
            
        spread_time = 0
        final_T_receiver = {}
        final_R_receiver = {}
        R_t_receiver_num = {0: len(seed_R_nodes)}
        search_range = SimpleQueue()
        
        candidate_node_dict = self.droped_auT_df.drop(seed_R_nodes+T_nodes).to_dict('index')
        new_gen_node_num = 0
        first_stop_time = 0

        # init final_T_receiver
        for node in self.authoritative_T.keys():
            final_T_receiver[node] = 1
            G.nodes[node]['group'] = 2
        for node in T_nodes:
            final_T_receiver[node] = 1
            G.nodes[node]['group'] = 2

        # init final_R_receiver & search_range
        spr_has_checked = {}
        for node in seed_R_nodes:
            final_R_receiver[node] = 1
            G.nodes[node]['group'] = 1
            G.nodes[node]['active_time'] = spread_time # record the active time of rumor node
            for nbr in nx.neighbors(G, node):
                if G.nodes[nbr]['group'] != 1:  # including T-active nodes
                    if nbr not in spr_has_checked:
                        search_range.put(nbr)
                        spr_has_checked[nbr]=1

        nothing_change = False
        is_pause = False  # if monitoring T is encountered, then pause the diffusion

        while (not nothing_change) and (not is_pause):

            nothing_change = True
            circulation_times = search_range.qsize()
            spread_time += 1

            for i in range(circulation_times):
                node = search_range.get()
                spr_has_checked.pop(node,0)
                if G.nodes[node]['group'] == 1: # This node is R-active (the search queue has two same nodes, the first node has been actived by rumor)
                    continue
                if G.nodes[node]['group'] == 2: # This node is T-active
                    is_pause = True
                    continue

                if self.__check_i_threshold(node, G) == 1:  # actived by rumor
                    nothing_change = False
                    G.nodes[node]['group'] = 1
                    G.nodes[node]['active_time'] = spread_time
                    final_R_receiver[node] = 1
                    candidate_node_dict.pop(node)

                    if not is_pause:  # avoid infinity adding nbr while 'is_pause' is true
                        for nbr in nx.neighbors(G, node):
                            if G.nodes[nbr]['group'] != 1:
                                if nbr not in spr_has_checked:
                                    search_range.put(nbr)
                                    spr_has_checked[nbr]=1
                else:
                    search_range.put(node)
                    spr_has_checked[node]=1
                    
            R_t_receiver_num[spread_time] = len(final_R_receiver)
            
            # the spreading stopped before detection
            if (nothing_change) and (not is_pause):
                # random select a node from candidate_node_df as R-node to continue spread
                new_gen_node = list(candidate_node_dict.keys())[randint(0,len(candidate_node_dict)-1)]
                candidate_node_dict.pop(new_gen_node)

                for nbr in nx.neighbors(G, new_gen_node):
                    if G.nodes[nbr]['group'] != 1:  # including T-active nodes
                        if nbr not in spr_has_checked:
                            search_range.put(nbr)
                            spr_has_checked[nbr]=1
                        
                G.nodes[new_gen_node]['group'] = 1
                G.nodes[new_gen_node]['active_time'] = spread_time
                final_R_receiver[new_gen_node] = 1
                R_t_receiver_num[spread_time] = len(final_R_receiver)
                nothing_change = False
                
                if first_stop_time == 0:
                    first_stop_time = spread_time
                new_gen_node_num +=1
        # the spreading stopped before detection
        # if not is_pause:
        #     R_t_receiver_num[spread_time+6] = len(final_R_receiver)
        #     spread_time+=6 # six degree of separation
            
        return G, spread_time, final_T_receiver, final_R_receiver, R_t_receiver_num

    def after_detected_diffusion(self,G:nx.Graph, spread_time:int, final_T_receiver:dict, final_R_receiver:dict, R_t_receiver_num:dict,T_node:list=None):
        
        # if not is_apply:
            # G = G.copy()
        final_T_receiver = final_T_receiver.copy()
        final_R_receiver = final_R_receiver.copy()
        R_t_receiver_num = R_t_receiver_num.copy()
        
        spr_search_range = SimpleQueue() # the node in this queue must be inactived
        cor_search_range = SimpleQueue() # the node in this queue must be R-actived

        if T_node is not None:
            for node in T_node:
                G.nodes[node]['group'] = 2
                final_T_receiver[node]=1
        
        spr_has_checked={}
        cor_has_checked={}
        # init spreading & correction search range
        for node in final_R_receiver.keys():
            for nbr in nx.neighbors(G,node):
                if G.nodes[nbr]['group'] == 0: # the nbr is inactive
                    if nbr not in spr_has_checked:
                        spr_search_range.put(nbr)
                        spr_has_checked[nbr]=1
        for node in final_T_receiver.keys():
            for nbr in nx.neighbors(G,node):
                if G.nodes[nbr]['group'] == 0: # the nbr is inactive
                    if nbr not in spr_has_checked:
                        spr_search_range.put(nbr)
                        spr_has_checked[nbr]=1
                elif G.nodes[nbr]['group'] == 1: # the nbr is R-active
                    if nbr not in cor_has_checked:
                        cor_search_range.put(nbr)
                        cor_has_checked[nbr]=1
        
        nothing_change = False
        
        while not nothing_change:
            nothing_change = True
            spr_circle_times = spr_search_range.qsize()
            cor_circle_times = cor_search_range.qsize()
            spread_time += 1

            # The phases of T & R spreading 
            for i in range(spr_circle_times):
                node = spr_search_range.get()
                spr_has_checked.pop(node,0)
                # if G.nodes[node]['group'] == 0:
                check_status = self.__check_i_threshold(node,G)
                if check_status == 1: # actived by rumor
                    nothing_change = False
                    G.nodes[node]['group'] = 1
                    G.nodes[node]['active_time'] = spread_time
                    final_R_receiver[node] = 1

                    for nbr in nx.neighbors(G,node): # update spread search range
                        if G.nodes[nbr]['group'] == 0:
                            if nbr not in spr_has_checked:
                                spr_search_range.put(nbr)
                                spr_has_checked[nbr]=1

                elif check_status == 2: # actived by truth
                    nothing_change = False
                    G.nodes[node]['group'] =  2
                    final_T_receiver[node] = 1

                    for nbr in nx.neighbors(G,node): # update spread search range
                        if G.nodes[nbr]['group'] == 0:
                            if nbr not in spr_has_checked:
                                spr_search_range.put(nbr)
                                spr_has_checked[nbr]=1
                        elif G.nodes[nbr]['group'] == 1: # if the nbr is R-active, then update correction search range
                            if nbr not in cor_has_checked:
                                cor_search_range.put(nbr)
                                cor_has_checked[nbr]=1
                else:
                    spr_search_range.put(node)
                    spr_has_checked[node]=1

            # The phases of correcting
            for i in range(cor_circle_times):
                node = cor_search_range.get()
                cor_has_checked.pop(node,0)

                if node in self.stubborn_R: # cannot correct the stubborn Rumor nodes, i.e. initial seed of Rumor nodes
                    continue

                if self.__check_c_threshold(node,G): # the rumor node is corrected successfully
                    G.nodes[node]['group'] = 2 
                    nothing_change = False
                    final_T_receiver[node] = 1
                    final_R_receiver.pop(node,0)

                    for nbr in nx.neighbors(G,node): # update spread search range
                        if G.nodes[nbr]['group'] == 0:
                            if nbr not in spr_has_checked:
                                spr_search_range.put(nbr)
                                spr_has_checked[nbr]=1
                        elif G.nodes[nbr]['group'] == 1: # if the nbr is R-active, then update correction search range
                            if nbr not in cor_has_checked:
                                cor_search_range.put(nbr)
                                cor_has_checked[nbr]=1
                else:
                    cor_search_range.put(node)
                    cor_has_checked[node]=1

            if not nothing_change:
                R_t_receiver_num[spread_time] = len(final_R_receiver)
        
        return G, spread_time, final_T_receiver, final_R_receiver, R_t_receiver_num
#%%
# ------------- test -------------------
# %%
