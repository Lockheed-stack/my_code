#%%
import networkx as nx
import numpy as np
# %%
class model_V2:

    def __init__(self, G: nx.Graph, is_init: bool = False, seed_T: list = None, seed_R: list = None) -> None:
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
                self.G.nodes[node]['i_threshold'] = np.random.uniform()
                self.G.nodes[node]['d_threshold'] = np.random.uniform()
                self.G.nodes[node]['status'] = 'inactive'
            # init R-seed nodes
            if len(seed_R) != 0:
                for node in seed_R:
                    self.G.nodes[node]['status'] = 'R-active'
                    self.final_R_receiver[node] = 'R'
            # init T-seed nodes
            if len(seed_T) != 0:
                for node in seed_T:
                    self.G.nodes[node]['status'] = 'T-active'
                    self.final_T_receiver[node] = 'T'
        else:
            for node in nx.nodes(G):
                if G.nodes[node]['status'] == 'R-active':
                    self.final_R_receiver[node] = 'R'
                elif G.nodes[node]['status'] == 'T-active':
                    self.final_T_receiver[node] = 'T'
            # init R-seed nodes
            if len(seed_R) != 0:
                for node in seed_R:
                    self.G.nodes[node]['status'] = 'R-active'
                    self.final_R_receiver[node] = 'R'
            # init T-seed nodes
            if len(seed_T) != 0:
                for node in seed_T:
                    self.G.nodes[node]['status'] = 'T-active'
                    self.final_T_receiver[node] = 'T'

    def diffusion(self, is_apply: bool = False):
        '''Simulation test of diffusion

        Parameters
        ---------------
        is_apply : bool
            Default False. If True, then this simulation will be applied on model's G.

        Return
        --------------
        (G ,time_step ,final_R_receiver , final_T_receiver, R_t_receiver) : tuple
            A Networkx's Graph which simulated diffusion has been completed. 
            time_step is total time the diffusion cost.
            final_R_receiver,final_T_receiver and  R_t_receiver are dict.
        '''
        if not is_apply:
            G = nx.Graph(self.G)
            final_T_receiver = self.final_T_receiver.copy()
            final_R_receiver = self.final_R_receiver.copy()

        else:
            G = self.G
            final_R_receiver = self.final_R_receiver
            final_T_receiver = self.final_T_receiver

        R_t_receiver = {0: [node for node in final_R_receiver.keys()]}  # Nodes activated by rumor at time t

        nothing_change = False
        # R_num = -1
        # T_num = -1
        time_step = 0

        # to decrease the search range
        i_search_range = {}
        d_search_range = {} # NOTE: It's init as '{}' because the initial R will not change to T.
        
        # Looking for potential node at active node surrounding
        for node in (list(final_R_receiver.keys())+list(final_T_receiver.keys())):
            for nbr in nx.neighbors(G,node):
                if G.nodes[nbr]['status'] == 'inactive':
                    i_search_range[nbr] = 1

        # while (R_num - len(final_R_receiver) or T_num - len(final_T_receiver)):
        while not nothing_change:
            time_step += 1

            nothing_change = True
            # R_num = len(final_R_receiver)
            # T_num = len(final_T_receiver)

            # influence stage
            for node in i_search_range.keys():
                if self.__check_i_threshold(node,G):
                    d_search_range[node] = 1
            # for node in nx.nodes(G):
            #     if G.nodes[node]['status'] == 'inactive':
            #         if self.__check_i_threshold(node, G):
            #             # nothing_change = False
            #             d_search_range[node]=1

            # decision stage
            # for node in nx.nodes(G):
            for node in list(d_search_range.keys()):
                if (G.nodes[node]['status'] == 'influenced') or (G.nodes[node]['status'] == 'R-active'):
                    if node not in self.seed_R:
                        flag = self.__check_d_threshold(node, G)
                        if flag != 0:
                            
                            if flag == 1:# decide to be R
                                if node not in final_R_receiver:
                                    final_R_receiver[node] = 'R'
                                    G.nodes[node]['status'] = 'R-active'
                                    G.nodes[node]['active_time'] = time_step  # Record the activation time
                                    #R_t_receiver[time_step] = [node for node in final_R_receiver.keys()]
                                    R_t_receiver[time_step] = list(final_R_receiver.keys())

                                    nothing_change = False
                            
                            elif flag == 2:# decide to be T
                                final_T_receiver[node] = 'T'
                                G.nodes[node]['status'] = 'T-active'
                                if node in final_R_receiver:
                                    final_R_receiver.pop(node)
                                    # R_t_receiver[time_step] = [node for node in final_R_receiver.keys()]
                                    R_t_receiver[time_step] = list(final_R_receiver.keys())
                                    
                                    d_search_range.pop(node)

                                    nothing_change = False
                            
                            if node in i_search_range:
                                i_search_range.pop(node)
                                for nbr in nx.neighbors(G,node):
                                    if G.nodes[nbr]['status'] == 'inactive':
                                        i_search_range[nbr]=1

        return G, time_step, final_R_receiver, final_T_receiver, R_t_receiver

    def __check_i_threshold(self, node, G: nx.Graph):
        influenced_num = 0
        node_deg = nx.degree(G, node)
        if node_deg == 0:
            return False

        for nbr in nx.neighbors(G, node):
            if (G.nodes[nbr]['status'] == 'R-active') or (G.nodes[nbr]['status'] == 'T-active'):
                influenced_num += 1

        if (influenced_num / node_deg) >= G.nodes[node]['i_threshold']:
            G.nodes[node]['status'] = 'influenced'
            return True

        return False

    def __check_d_threshold(self, node, G: nx.Graph):
        active_num = 0
        R_active_num = 0
        # T_active_num = 0

        if (G.nodes[node]['status'] == 'T-active'):# T-active node cannot change to be R
            return 0

        for nbr in nx.neighbors(G, node):
            if G.nodes[nbr]['status'] == 'R-active':
                active_num += 1
                R_active_num += 1
            elif G.nodes[nbr]['status'] == 'T-active':
                active_num += 1
                # T_active_num += 1

        if active_num == 0:
            return 0

        # decide to be R
        elif (R_active_num / active_num) > G.nodes[node]['d_threshold']:
            return 1
        # decide to be T
        else:
            return 2
        # elif T_active_num > 0:
        #     return 2

        # return 0

    def update_seed_T(self, seed_T: list, override: bool = False):
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
        Only inactive node will be changed when override is False. If status of node is R-active or T-active, it will
        omit this update.
        '''

        if (seed_T is None):
            return

        if not override:
            self.seed_T += seed_T

            for node in seed_T:
                if self.G.nodes[node]['status'] == 'inactive':
                    self.G.nodes[node]['status'] = 'T-active'
                    self.final_T_receiver[node] = 'T'
        else:
            for node in self.seed_T:
                self.G.nodes[node]['status'] = 'inactive'
                if node in self.final_T_receiver:
                    self.final_T_receiver.pop(node)
            for node in seed_T:
                self.G.nodes[node]['status'] = 'T-active'
                self.final_T_receiver[node] = 'T'
            self.seed_T = seed_T.copy()

    def update_seed_R(self, seed_R: list, override: bool = False):
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
        Only inactive node will be changed when override is False. If status of node is R-active or T-active, it will
        omit this update.
        '''

        if (seed_R is None):
            return

        if not override:
            self.seed_R += seed_R

            for node in seed_R:
                if self.G.nodes[node]['status'] == 'inactive':
                    self.G.nodes[node]['status'] = 'R-active'
                    self.final_R_receiver[node] = 'R'
        else:
            for node in self.seed_R:
                self.G.nodes[node]['status'] = 'inactive'
                if node in self.final_R_receiver:
                    self.final_R_receiver.pop(node)
            for node in seed_R:
                self.G.nodes[node]['status'] = 'R-active'
                self.final_R_receiver[node] = 'R'
            self.seed_R = seed_R.copy()

    def get_initialized_G(self,):
        return nx.Graph(self.G)

    def copy(self,):
        G = nx.Graph(self.G)
        return self.__class__(G, True, self.seed_T.copy(), self.seed_R.copy())


# %%
