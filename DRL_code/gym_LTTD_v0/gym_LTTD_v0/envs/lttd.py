from math import exp,log
import gymnasium as gym
from gymnasium import spaces

import networkx as nx
import pandas as pd
from random import uniform, randint, sample
from queue import SimpleQueue
import numpy as np


class lttdENV(gym.Env):
    def __init__(
        self,
        G: nx.Graph,
        init_rumor_rate: float = 0.05,
        au_T_rate: float = 0.01,
        k_budget: float = 0.01,
        target_to_reach: float = 0,
        alpha: float = 0.8,
        is_for_test:bool=False
    ) -> None:
        super().__init__()

        self.G = G.copy()
        self.init_rumor_rate = init_rumor_rate
        self.au_T_rate = au_T_rate
        self.nodes_num = self.G.number_of_nodes()
        self.select_k_Tnodes = round(k_budget * self.nodes_num)
        if self.select_k_Tnodes <= 0:
            self.select_k_Tnodes = 1
        self.target_to_reach = target_to_reach
        self.alpha = alpha
        self.is_for_test = is_for_test

        self.sorted_deg = sorted(
            dict(self.G.degree()).items(), key=lambda x: x[1], reverse=True
        )
        self.set_G_nodes = set(G.nodes())
        self.df = pd.DataFrame(
            self.G.degree, columns=["node", "degree"], dtype=int
        ).set_index("node")
        self.median_degree = self.df["degree"].median()
        self.df_node_index = pd.DataFrame(
            self.G.nodes(), columns=["node"], dtype=int
        )  # the original nodes may be disordered

        self.stubborn_R = set()  # the seed of rumor nodes
        self.final_T_receiver = self._select_authoritative_T_nodes(au_T_rate)
        self.final_R_receiver = set()
        self.recommend_nodes = set()
        self.actual_k_select = 0

        """
        group:
            inactive:0
            R-active:1
            T-active:2
        """
        self.observation_space = spaces.Box(
            0, 2, shape=(self.nodes_num,), dtype=np.int32
        )
        """
        the num of actions equal to the number of nodes
        """
        self.action_space = spaces.Discrete(self.nodes_num)

        for node in nx.nodes(G):
            self.G.nodes[node]["i_threshold"] = uniform(
                0, 1)  # influenced threshold
            self.G.nodes[node]["c_threshold"] = uniform(
                0, 1)  # correction threshold
            self.G.nodes[node]["group"] = 0

        # update new correction threshold
        for node in self.final_T_receiver:
            self.G.nodes[node]["group"] = 2
            self._new_correction_threshold(node)

        # before detected diffusion
        self._before_detected_diffusion(
            self._generate_R_nodes(self.init_rumor_rate))
        self.rest_avail_nodes_num = len(self.set_G_nodes-self.final_R_receiver-self.final_T_receiver)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.stubborn_R = set()
        self.final_T_receiver = self._select_authoritative_T_nodes(
            self.au_T_rate)
        self.final_R_receiver = set()
        self.actual_k_select = 0
        self.recommend_nodes = set()
        for node in nx.nodes(self.G):
            # influenced threshold
            self.G.nodes[node]["i_threshold"] = uniform(0, 1)
            # correction threshold
            self.G.nodes[node]["c_threshold"] = uniform(0, 1)
            self.G.nodes[node]["group"] = 0

        # update new correction threshold
        for node in self.final_T_receiver:
            self.G.nodes[node]["group"] = 2
            self._new_correction_threshold(
                node,
            )

        # before detected diffusion
        self._before_detected_diffusion(
            self._generate_R_nodes(self.init_rumor_rate))
        self.rest_avail_nodes_num = len(self.set_G_nodes-self.final_R_receiver-self.final_T_receiver)
        observation = self._get_obs()
        info = self._get_info()

        return np.array(observation, dtype=np.int32), info

    def step(self, action: int):
        # the 'action' parameret is the index of real action. it's need to be transform.
        action = self.df_node_index.iloc[action]["node"]
        self.G.nodes[action]["group"] = 2
        self.final_T_receiver.add(action)
        self.actual_k_select += 1
        self.recommend_nodes.discard(action)
        # It will not actually diffuse untill k T nodes are selected.
        if (self.actual_k_select + 1 <= self.select_k_Tnodes):
            truncated = (
                len(self.final_R_receiver | self.final_T_receiver) >= self.nodes_num
            )
            if (truncated):  # truncated if there is no any node could be selected after this action
                return self._after_detected_diffusion()

            observation = self._get_obs()
            info = self._get_info()
            terminated = False
            if self.is_for_test:
                reward = 0
            else:
                # reward = self._reward_shaping(action)
                reward = 1
            return (
                np.array(observation, dtype=np.int32),
                reward,
                terminated,
                truncated,
                info,
            )

        # the last action is choosen, then the diffusion start.
        return self._after_detected_diffusion()

    def _get_obs(self):
        return list(dict(self.G.nodes(data="group", default=0)).values())

    def _get_info(self):
        return {"T_active": self.final_T_receiver, "R_active": self.final_R_receiver, "recommend":self.recommend_nodes}

    def _new_correction_threshold(self, au_node, beta: float = 1, n_th_nbr: int = 3):
        """Updating correction thresholds for nodes which under the influence by au_T nodes.

        Parameters
        ----------
        au_node : inferr
            The au_T node.
        beta : float, optional
            The power of au_T node's influence, by default 1. The smaller beta is, the more
            powerful ability of correction of au_T node.
        n_th_nbr : int, optional
            The range of au_T nodes'influence, by default 3.
        """
        nbr_queue = SimpleQueue()
        checked_node = {au_node: 1}

        for nbr in nx.neighbors(self.G, au_node):
            nbr_queue.put((nbr, 1))  # (node,k-th nbr)
            checked_node[nbr] = 1

        while not nbr_queue.empty():
            node, order = nbr_queue.get()

            # update new correction threshold
            p = self.G.nodes[node]["c_threshold"]
            self.G.nodes[node]["c_threshold"] = p - \
                p / (1 + exp(beta * (order)))

            if order < n_th_nbr:
                for nbr in nx.neighbors(self.G, node):
                    if nbr in checked_node:
                        continue
                    nbr_queue.put((nbr, order + 1))
                    checked_node[nbr] = 1

    def _select_authoritative_T_nodes(self, T_percent: float):
        # median_degree = self.df["degree"].median()

        ge_median_nodes = self.df.query(f"degree>={self.median_degree}")

        if (ge_median_nodes.shape[0] * T_percent) < 1:
            T_sample_df = ge_median_nodes.sample(1, replace=False)
        else:
            T_sample_df = ge_median_nodes.sample(frac=T_percent, replace=False)

        T_nodes = T_sample_df.to_dict("index")

        return set(T_nodes)

    def _check_i_threshold(self, node: int):
        """check influence threshold

        Parameters
        ----------
        node : nx.Graph node
            a nx.Graph node

        Returns
        -------
        int
            0 : nothing change
            1 : activated by rumor
            2 : activated by truth
        """
        # influence_num = 0
        T_num, R_num = 0, 0
        node_deg = nx.degree(self.G, node)
        if node_deg == 0:
            return 0

        for nbr in nx.neighbors(self.G, node):
            if self.G.nodes[nbr]["group"] != 0:
                # influence_num +=1
                if self.G.nodes[nbr]["group"] == 1:
                    R_num += 1
                else:
                    T_num += 1

        # Priority to become a rumor node
        if (R_num / node_deg) >= self.G.nodes[node]["i_threshold"]:
            return 1
        elif (T_num / node_deg) >= self.G.nodes[node]["i_threshold"]:
            return 2

        return 0

    def _check_c_threshold(self, node: int):
        actived_num = 0
        T_active_num = 0

        for nbr in nx.neighbors(self.G, node):
            check_status = self.G.nodes[nbr]["group"]

            if check_status != 0:
                actived_num += 1
            if check_status == 2:
                T_active_num += 1

        if actived_num == 0:
            return False
        elif (T_active_num / actived_num) >= self.G.nodes[node]["c_threshold"]:
            return True

        return False

    def _generate_R_nodes(self, R_rate: float = 0.05):
        inactive_nodes = list(self.set_G_nodes - self.final_T_receiver)
        if (R_rate * len(inactive_nodes)) < 1:
            R_nodes = sample(inactive_nodes, 1)
        else:
            cnt = round(R_rate * len(inactive_nodes))
            R_nodes = sample(inactive_nodes, cnt)

        self.stubborn_R.clear()
        for node in R_nodes:
            self.stubborn_R.add(node)
            self.final_R_receiver.add(node)
            self.G.nodes[node]["group"] = 1

        return R_nodes

    def _before_detected_diffusion(self, seed_R_nodes: list):
        """Simulation test of diffusion

        Parameters
        ----------
        seed_R_nodes : list
            the rumor seed nodes
        """
        search_range = SimpleQueue()
        candidate_node_dict = self.set_G_nodes - \
            self.stubborn_R - self.final_T_receiver

        # init final_R_receiver & search_range
        spr_has_checked = set()
        for node in seed_R_nodes:
            # self.final_R_receiver.add(node)
            # self.G.nodes[node]["group"] = 1
            for nbr in nx.neighbors(self.G, node):
                # including T-active and inactive nodes
                if (self.G.nodes[nbr]["group"] != 1):
                    if nbr not in spr_has_checked:
                        search_range.put(nbr)
                        spr_has_checked.add(nbr)

        nothing_change = False
        is_pause = False  # if monitoring T is encountered, then pause the diffusion

        while (not nothing_change) and (not is_pause):
            nothing_change = True
            circulation_times = search_range.qsize()

            for _ in range(circulation_times):
                node = search_range.get()
                spr_has_checked.discard(node)

                # This node is R-active (the search queue has two same nodes, the first node has been actived by rumor)
                if self.G.nodes[node]["group"] == 1:
                    continue
                # This node is T-active
                if self.G.nodes[node]["group"] == 2:
                    is_pause = True
                    continue

                if self._check_i_threshold(node) == 1:  # actived by rumor
                    nothing_change = False
                    self.G.nodes[node]["group"] = 1
                    self.final_R_receiver.add(node)
                    candidate_node_dict.discard(node)

                    if (not is_pause):  # avoid infinity adding nbr while 'is_pause' is true
                        for nbr in nx.neighbors(self.G, node):
                            if self.G.nodes[nbr]["group"] != 1:
                                if nbr not in spr_has_checked:
                                    search_range.put(nbr)
                                    spr_has_checked.add(nbr)
                else:
                    search_range.put(node)
                    spr_has_checked.add(node)

            # the spreading stopped before detection
            if (nothing_change) and (not is_pause):
                # random select a node from candidate_node_df as R-node to continue spread
                new_gen_node = list(candidate_node_dict)[
                    randint(0, len(candidate_node_dict) - 1)
                ]
                candidate_node_dict.discard(new_gen_node)

                for nbr in nx.neighbors(self.G, new_gen_node):
                    # including T-active and inactive nodes
                    if (self.G.nodes[nbr]["group"] != 1):
                        if nbr not in spr_has_checked:
                            search_range.put(nbr)
                            spr_has_checked.add(nbr)

                self.G.nodes[new_gen_node]["group"] = 1
                self.final_R_receiver.add(new_gen_node)
                nothing_change = False
        
        # update recommend nodes
        if not self.is_for_test:
            for node in self.final_R_receiver:
                for nbr in nx.neighbors(self.G,node):
                    if self.G.nodes[nbr]['group'] == 0:
                        self.recommend_nodes.add(nbr)

    def _after_detected_diffusion(self,):
        # the last action before after_detected_diffusion
        spr_search_range = SimpleQueue()  # the node in this queue must be inactived
        cor_search_range = SimpleQueue()  # the node in this queue must be R-actived

        spr_has_checked = set()
        cor_has_checked = set()

        for node in self.final_R_receiver:
            for nbr in nx.neighbors(self.G, node):
                if self.G.nodes[nbr]["group"] == 0:  # the nbr is inactive
                    if nbr not in spr_has_checked:
                        spr_search_range.put(nbr)
                        spr_has_checked.add(nbr)
        for node in self.final_T_receiver:
            for nbr in nx.neighbors(self.G, node):
                if self.G.nodes[nbr]["group"] == 0:  # the nbr is inactive
                    if nbr not in spr_has_checked:
                        spr_search_range.put(nbr)
                        spr_has_checked.add(nbr)
                elif self.G.nodes[nbr]["group"] == 1:  # the nbr is R-active
                    if nbr not in cor_has_checked:
                        cor_search_range.put(nbr)
                        cor_has_checked.add(nbr)

        # start spreading and correcting
        nothing_change = False
        R_start_num = len(self.final_R_receiver)
        T_start_num = len(self.final_T_receiver)
        while not nothing_change:
            nothing_change = True
            spr_circle_times = spr_search_range.qsize()
            cor_circle_times = cor_search_range.qsize()

            # The phases of T & R spreading
            for _ in range(spr_circle_times):
                node = spr_search_range.get()
                spr_has_checked.discard(node)

                check_status = self._check_i_threshold(node)
                if check_status == 1:  # actived by rumor
                    nothing_change = False
                    self.G.nodes[node]["group"] = 1
                    self.final_R_receiver.add(node)

                elif check_status == 2:  # actived by truth
                    nothing_change = False
                    self.G.nodes[node]["group"] = 2
                    self.final_T_receiver.add(node)
                else:  # still inactive
                    nothing_change = True
                    spr_search_range.put(node)
                    spr_has_checked.add(node)
                    continue

                for nbr in nx.neighbors(self.G, node):
                    # update search range
                    if self.G.nodes[nbr]["group"] == 0:
                        if nbr not in spr_has_checked:
                            spr_has_checked.add(nbr)
                            spr_search_range.put(nbr)
                    # update correction search range
                    elif check_status == 2 and self.G.nodes[nbr]["group"] == 1:
                        if nbr not in cor_has_checked:
                            cor_search_range.put(nbr)
                            cor_has_checked.add(nbr)

            # The phases of correcting
            for _ in range(cor_circle_times):
                node = cor_search_range.get()
                cor_has_checked.discard(node)

                # cannot correct the stubborn Rumor nodes, i.e. initial seed of Rumor nodes
                if (node in self.stubborn_R):
                    continue
                # the rumor node is corrected successfully
                if self._check_c_threshold(node):
                    self.G.nodes[node]["group"] = 2
                    self.final_T_receiver.add(node)
                    self.final_R_receiver.discard(node)
                    nothing_change = False
                    for nbr in nx.neighbors(self.G, node):
                        # update spread search range
                        if (self.G.nodes[nbr]["group"] == 0 and nbr not in spr_has_checked):
                            spr_search_range.put(nbr)
                            spr_has_checked.add(nbr)
                        # update correction search range
                        elif (self.G.nodes[nbr]["group"] == 1 and nbr not in cor_has_checked):
                            cor_search_range.put(nbr)
                            cor_has_checked.add(nbr)
                else:
                    cor_search_range.put(node)
                    cor_has_checked.add(node)

        observation = self._get_obs()
        info = self._get_info()
        terminated = nothing_change

        if self.is_for_test:
            reward = 0
        else:
            # RN_rate = len(self.final_R_receiver) / self.nodes_num
            TR_rate = len(self.final_T_receiver) / len(self.final_R_receiver)

            delta_R = (len(self.final_R_receiver)-R_start_num)
            delta_T = (len(self.final_T_receiver)-T_start_num)

            reward = delta_T-delta_R

            if reward > 0:
                # if RN_rate <= self.target_to_reach:
                #     reward *= 2.3
                if TR_rate >= self.target_to_reach:
                    reward *= 3
                elif TR_rate > max(self.target_to_reach/2,1):
                    reward *= 2.2

        return np.array(observation, dtype=np.int32), reward, terminated, False, info

    def _reward_shaping(self, action: int):
        
        R_around_num = 0
        # rough estimate influence
        for nbr in self.G.neighbors(action):
            if self.G.nodes[nbr]['group'] == 1:
                R_around_num +=1
        
        reward = R_around_num + (self.G.degree(action)*self.alpha)

        return reward
        # checked_node = set()
        # for node in (self.final_R_receiver|self.final_T_receiver):
        #     for nbr in self.G.neighbors(node):
        #         if nbr not in checked_node:
        #             checked_node.add(nbr)
        #             if self.G.nodes[nbr]['group'] ==1:
        #                 if self._check_c_threshold(nbr):
        #                     delta_R-=1
        #                     delta_T+=1
        #             elif self.G.nodes[nbr]['group']==0:
        #                 check_res = self._check_i_threshold(nbr)
        #                 if check_res ==2:
        #                     delta_T+=1
        #                 elif check_res == 1:
        #                     delta_R+=1

        # return delta_T-delta_R
