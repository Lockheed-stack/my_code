#%%
import networkx as nx
import numpy as np
import LTD1DT
# %%

def MinGreedy(model:LTD1DT.model_V2, seed_R: list = None, k: int = 1):
    '''Using greedy algorithm to find k truth nodes.

    Parameters
    ----------
    model : LTD1DT.model_v2
        A LTD1DT model
    seed_R : list, optional
        A list of rumor nodes, by default None
    k : int, optional
        To decide how many truth nodes will be chosen, by default 1

    Returns
    -------
    seed_T : list
        A list of k truth nodes.
    '''
    seed_T = []
    if seed_R is None:
        print('seed R has not  given.')
        return seed_T
    if k <= 0:
        # print('invaild k.')
        return seed_T

    # test_model = LTD1DT.model_V2(G,False,[],seed_R)
    test_model = model.copy()
    test_model.update_seed_R(seed_R, True)

    G, time_step, final_R, final_T, R_t_receiver = test_model.diffusion()
    final_R_num = len(final_R)

    while (len(seed_T) < k):

        chosen_node = None
        alternative_node = None
        temp_model = test_model.copy()

        for node in nx.nodes(G):
            if (node not in seed_T) and (node not in seed_R):
                temp_model.update_seed_T(seed_T + [node], True)
                result = temp_model.diffusion() # 99.7% of the time is spent there.
                if final_R_num > len(result[2]):
                    chosen_node = node
                    final_R_num = len(result[2])
                elif final_R_num == len(result[2]):
                    alternative_node = node

            temp_model.update_seed_T(seed_T, True)

        if chosen_node is not None:
            seed_T.append(chosen_node)
        elif alternative_node is not None:
            seed_T.append(alternative_node)
        else:
            print(f'Cannot find {k} T nodes, only {len(seed_T)} found.')
            return seed_T

    return seed_T

def pagerank(G: nx.Graph, seed_R:list=None,k: int = 0, alpha: float = 0.85, theta: float = 1e-6, iter_num: int = 100):
    '''Calculating the Pagerank value of G

    Parameters
    ------------
    G: nx.Graph
        A Networkx Graph

    k: int
        Choose top k nodes as Truth seed. If (k < 0) or (k > node num), it will return sorted and descend pagerank value.

    alpha: float, optional
        Damping parameter for PageRank, by default=0.85.

    theta: float, optional
        The threshold of the pagerank value that coveraged to a rational value.
        Default = 1e-6

    iter_num: int, optional
        Specifying a max iteration times. Default is None.

    Returns
    ---------
    seed_T: list
        Top k pagerank value Nodes

    Note
    ---------------
    The iteration will stop after an error tolerance of len(G) * theta has been reached. 
    If the number of iterations exceed max_iter, 
    a PowerIterationFailedConvergence exception is raised.
    '''

    if k == 0:
        return []

    # it's an adjacency matrix now
    node_list = list(nx.nodes(G))
    # trans_mat = nx.adjacency_matrix(G, dtype=np.float64).toarray()
    trans_mat = nx.to_numpy_matrix(G,dtype=np.float64) # return numpy.matrix
    n = nx.number_of_nodes(G)

    # switch to transition probability matrix
    for i in range(n):
        trans_mat[:, i] = trans_mat[:, i] / np.dot(np.ones((1, n)), trans_mat[:, i])

    # pagerank vector(column)
    p = np.ones((n, 1),) / n
    trans_mat = trans_mat.A
    seed_T = {}

    for _ in range(iter_num):
        p_last = p.copy()
        p = (alpha * (np.dot(trans_mat, p))) + ((alpha / n) * np.ones((n, 1)))

        # check coveragence, L1 norm
        err = np.absolute(p - p_last).sum()

        if err < (n * theta):
            
            # choose top k nodes
            if (len(p) <= k) or (k<0):
                # return [node for node in nx.nodes(G)]
                node_pg_dict = {}
                for i in range(nx.number_of_nodes(G)):
                    if node_list[i] not in seed_R:
                        node_pg_dict[node_list[i]] = p[i]
                sorted_pg = sorted(node_pg_dict.items(),key=lambda x:x[1],reverse=True)
                return list(dict(sorted_pg).keys())

            else:
                reshape_p = np.reshape(p, np.shape(p)[0])
                while (k):
                    max_pValue = 0
                    chosen_node = 0
                    for i in range(np.shape(reshape_p)[0]):
                        if reshape_p[i] > max_pValue:
                            if ((node_list[i]) not in seed_T) and ((node_list[i]) not in seed_R):
                                max_pValue = reshape_p[i]
                                chosen_node = node_list[i]
                    seed_T[chosen_node] = max_pValue
                    k -= 1
                # return seed_T
                return list(seed_T.keys())

    raise nx.PowerIterationFailedConvergence(iter_num)

def ContrId(model: LTD1DT.model_V2, seed_R: list = None, k: int = 1):
    '''Contributors Identification

    Parameters:
    ---------------
    model: LTD1DT.model_V2
        A LTD1DT model.
    seed_R: list
        The seed nodes of rumor
    k: int 
        To decide how many truth node will be chosen. Default = 1.

    Return:
    ----------------
    seed_T : list
        A list of seed_T of top k
    Note:
    ------------------
        This function will try to meet the requirement of finding k truth nodes.
        So the return value may exsist some nodes that have same contribution.
    '''

    if k <= 0:
        return []
    # if G is None:
    #     print('a Networkx Graph need to be given.')
    #     return
    if seed_R is None:
        print('please give a seed_R')
        return
    else:
        temp_model = model.copy()
        temp_model.update_seed_T([], override=True)
        temp_model.update_seed_R(seed_R, override=True)
        res = temp_model.diffusion()
        # temp_model = LTD1DT.model_V2(G,False,[],seed_R)
        # res = temp_model.diffusion()

    seed_T = {} # As return value
    node_ctr = {}
    for node in nx.get_node_attributes(res[0], 'active_time').items():
        ctr = 0  # contribution
        for nbr in nx.neighbors(res[0], node[0]):
            if nbr in nx.get_node_attributes(res[0], 'active_time'):
                if res[0].nodes[nbr]['active_time'] > node[1]:
                    ctr += 1
        node_ctr[node[0]] = ctr


    if k > len(node_ctr):
        # print(f'cannot find {k} truth nodes, only {len(node_ctr)} nodes found.')
        return list(node_ctr.keys())
        

    else:
        while (k):
            max_ctr = 0
            chosen_node = None
            alternative_node = None

            for node in node_ctr.items():
                if node[1] > max_ctr:
                    chosen_node = node[0]
                    max_ctr = node[1]
                elif node[1] == max_ctr:
                    alternative_node = node[0]

            if chosen_node is not None:
                seed_T[chosen_node] = max_ctr
                node_ctr.pop(chosen_node)
                k -= 1
            elif alternative_node is not None:
                seed_T[alternative_node] = max_ctr
                node_ctr.pop(alternative_node)
                k -= 1
            else:
                print(f'cannot find another {k} more truth nodes, only {len(seed_T)} nodes found.')
                # return seed_T
                return list(seed_T.keys())

    # return seed_T
    return list(seed_T.keys())


# %%
# G_scale_free  = nx.barabasi_albert_graph(500,1)
# %load_ext line_profiler
# %lprun -f MinGreedy MinGreedy(G_scale_free,10,[0,3,7])
#%%
# %time list(a.keys()) # winner
# %time [node for node in a.keys()]
# %%
