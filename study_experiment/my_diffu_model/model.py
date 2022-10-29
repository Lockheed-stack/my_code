#%%
import networkx as nx
import numpy as np
import pandas as pd
import queue


#%%
class model:

    def __init__(self, G: nx.Graph, au_T_rate: float = 0.1, org_R_rate: float = 0.1,) -> None:
        '''Creating my model, improvement based on LT model

        Parameters
        ----------
        G : nx.Graph
            A Networkx's Graph.
        au_T_rate : float
            The percentage of authoritative Truth node.
        org_R_rate : float
            The percentage of organized Rumor node.
        '''

        self.G = G.copy()

        # these nodes will not change their status
        # self.authoritative_T,self.organized_R = self._select_T_R_node(G,au_T_rate,org_R_rate)

        self.authoritative_T = self.select_authoritative_T_nodes(au_T_rate)
        self.stubborn_R = []
        '''
        status:
            inactive:0
            R-active:1
            T-active:2
        '''
        for node in nx.nodes(G):
            self.G.nodes[node]['i_threshold'] = np.random.uniform()  # influenced threshold
            self.G.nodes[node]['c_threshold'] = np.random.uniform()  # correction threshold
            self.G.nodes[node]['status'] = 0

        # update new correction threshold
        for node in self.authoritative_T:
            self.G.nodes[node]['status'] = 2
            self.__new_correction_threshold(node,)

    def __new_correction_threshold(self, au_node, beta: float = 1, n_th_nbr: int = 4):
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
        nbr_queue = queue.Queue()
        checked_node = {au_node: 1}

        for nbr in nx.neighbors(self.G, au_node):
            nbr_queue.put((nbr, 1))  # (node,k-th nbr)
            checked_node[nbr] = 1

        while not nbr_queue.empty():
            node, order = nbr_queue.get()

            # update new correction threshold
            p = self.G.nodes[node]['c_threshold']
            self.G.nodes[node]['c_threshold'] = p - p / (1 + np.exp(beta * (order)))

            if order < n_th_nbr:
                for nbr in nx.neighbors(self.G, node):
                    if nbr in checked_node:
                        continue
                    nbr_queue.put((nbr, order + 1))
                    checked_node[nbr] = 1

    def __select_T_R_node(self, G: nx.Graph, T_percent: float, R_percent: float):

        df = pd.DataFrame(G.degree, columns=['node', 'degree'], dtype=int).set_index('node')
        median_degree = np.median(list(dict(G.degree).values()))

        ge_median_node = df.query(f'degree > {median_degree}')

        T_sample_df = ge_median_node.sample(frac=T_percent, replace=False)
        T_nodes = list(T_sample_df.index)

        R_sample_df = ge_median_node.drop(T_sample_df.index,)
        R_nodes = list(R_sample_df.sample(frac=R_percent, replace=False).index)

        return T_nodes, R_nodes

    def select_authoritative_T_nodes(self, T_percent: float):

        df = pd.DataFrame(self.G.degree, columns=['node', 'degree'], dtype=int).set_index('node')
        median_degree = df['degree'].median()

        ge_median_nodes = df.query(f'degree>{median_degree}')

        if ge_median_nodes.shape[0] < 10:
            T_sample_df = ge_median_nodes.sample(1, replace=False)
        else:
            T_sample_df = ge_median_nodes.sample(frac=T_percent, replace=False)

        T_nodes = list(T_sample_df.index)

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
            if G.nodes[nbr]['status'] != 0:
                #influence_num +=1
                if G.nodes[nbr]['status'] == 1:
                    R_num += 1
                else:
                    T_num += 1

        # Priority to become a rumor node
        if (R_num / node_deg) >= G.nodes[node]['i_threshold']:
            return 1
        elif (T_num / node_deg) >= G.nodes[node]['i_threshold']:
            return 2

        return 0

    def generate_R_nodes(self, org_R_num: int, normal_R_num: int, additional_T_node: list):
        if normal_R_num <= 0:
            return []

        df = pd.DataFrame(self.G.degree, columns=['node', 'degree'], dtype=int).set_index('node')

        # drop node which in authoritative T node list
        au_dropped_df = df.drop(self.authoritative_T,)

        median_degree = df['degree'].median()
        ge_median_nodes = au_dropped_df.query(f'degree>{median_degree}')

        # select organized rumor seed nodes
        org_R_node = list(ge_median_nodes.sample(org_R_num, replace=False).index)

        # select normal rumor seed nodes
        org_au_dropped_df = au_dropped_df.drop(org_R_node)
        normal_R_node = list(org_au_dropped_df.sample(normal_R_num, replace=False).index)

        return org_R_node + normal_R_node

    def update_seed_R_status(self, seed_R: list, G: nx.Graph):
        if seed_R is not None:
            for node in seed_R:
                G.nodes[node]['status'] = 1

    def refresh_i_c_threshold(self,new_au_T:bool=False,au_T_rate:float=0.0):
        
        if new_au_T:
            self.authoritative_T = self.select_authoritative_T_nodes(au_T_rate)
        
        for node in nx.node(self.G):
            self.G.nodes[node]['i_threshold'] = np.random.uniform()  # influenced threshold
            self.G.nodes[node]['c_threshold'] = np.random.uniform()  # correction threshold
            self.G.nodes[node]['status'] = 0
        
        for node in self.authoritative_T:
            self.G.nodes[node]['status'] = 2
            self.__new_correction_threshold(node,)


    def before_detected_diffusion(self, seed_R_nodes: list, T_nodes: list, is_apply: bool = False):
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
            G = self.G.copy()
        else:
            G = self.G

        spread_time = 0
        final_T_receiver = {}
        final_R_receiver = {}
        R_t_receiver_num = {0: len(seed_R_nodes)}
        search_range = queue.Queue()

        # init final_T_receiver
        for node in (self.authoritative_T + T_nodes):
            final_T_receiver[node] = 1
            G.nodes[node]['status'] = 2

        # init final_R_receiver & search_range
        self.update_seed_R_status(seed_R_nodes, G)

        for node in seed_R_nodes:
            final_R_receiver[node] = 1
            G.nodes[node]['status'] = 1
            for nbr in nx.neighbors(G, node):
                if G.nodes[nbr]['status'] != 1:  # including T-active nodes
                    search_range.put(nbr)

        nothing_change = False
        is_pause = False  # if meet monitoring T, then pause the diffusion

        while not nothing_change:

            nothing_change = True
            circulation_times = search_range.qsize()

            if not is_pause:
                spread_time += 1
                for i in range(circulation_times):
                    node = search_range.get()
                    if G.nodes[node]['status'] == 2:
                        is_pause = True
                        continue

                    if self.__check_i_threshold(node, G) == 1:  # actived by rumor
                        nothing_change = False
                        G.nodes[node]['status'] = 1
                        final_R_receiver[node] = 1

                        if not is_pause:  # avoid infinity adding nbr while 'is_pause' is true
                            for nbr in nx.neighbors(G, node):
                                if G.nodes[nbr]['status'] != 1:
                                    search_range.put(nbr)
                R_t_receiver_num[spread_time] = len(final_R_receiver)
            else:
                break

        return G, spread_time, final_T_receiver, final_R_receiver, R_t_receiver_num

    def after_detected_diffusion(self,G:nx.Graph, spread_time:int, final_T_receiver:dict, final_R_receiver:dict, R_t_receiver_num:dict):
        pass
#%%
# ------------- test -------------------
if __name__ == '__main__':
    model1 = model(nx.karate_club_graph(),)
#%%
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (1, 5), (1, 7), (2, 4), (2, 6), (3, 4), (3, 8), (3, 5), (5, 9), (7, 10), (6, 8),
                  (10, 9), (11, 1), (9, 1)])
nx.draw_networkx(G, pos=nx.circular_layout(G), with_labels=True)
# %%
model1.G.degree
#%%
model1.authoritative_T
# %%
r_node = model1.generate_R_nodes(1, 5, [])
r_node
#%%
model1.before_detected_diffusion(
    r_node,
    [],
)
# %%
nx.get_node_attributes(model1.G, 'i_threshold')
# %%
model1.update_seed_R_status([3], G)
