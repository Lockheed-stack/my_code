#%%
import networkx as nx
import numpy as np
import LTD1DT
import scipy.sparse
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

        G , final_R , final_T, R_t_receiver = test_model.diffusion()
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

    def pagerank(self,G:nx.Graph,alpha:float=0.85,theta:float=1e-6,iter_num:int=100):
        '''Calculating the Pagerank value of G

        Parameters
        ------------
        G: nx.Graph
            A Networkx Graph
        alpha: float, optional
            Damping parameter for PageRank, default=0.85.
        theta: float, optional
            The threshold of the pagerank value that coveraged to a rational value.
            Default = 1e-6
        iter_num: int, optional
            Specifying a max iteration times. Default is None.

        Returns
        ---------
        p: np.array
            Nodes pagerank value

        Note
        ---------------
        The iteration will stop after an error tolerance of len(G) * theta has been reached. 
        If the number of iterations exceed max_iter, 
        a PowerIterationFailedConvergence exception is raised.
        '''
        # it's an adjacency matrix now
        node_list = [i for i in range(1,nx.number_of_nodes(G)+1)]
        trans_mat = nx.adjacency_matrix(G,nodelist=node_list,dtype=np.float64).toarray()
        n = nx.number_of_nodes(G)

        # switch to transition probability matrix
        for i in range(n):
            trans_mat[:,i] = trans_mat[:,i]/np.dot(np.ones((1,n)),trans_mat[:,i])

        # pagerank vector(column)
        p = np.ones((n,1),)/n

        for _ in range(iter_num):
            p_last = p.copy()
            p = (alpha*(np.dot(trans_mat,p))) + ((alpha/n)*np.ones((n,1)))

            # check coveragence, L1 norm
            err = np.absolute(p-p_last).sum()

            if err < (n*theta):
                return p
        raise nx.PowerIterationFailedConvergence(iter_num)

    def ContrId(self,k:int=1,model:LTD1DT.model_V2=None):
        '''Contributors Identification

        Parameters:
        ---------------
        k: int 
            To decide how many truth node will be chosen. Default = 1.
        model: LTD1DT.model_V2
            A LTD1DT model.

        Return:
        ----------------
        seed_T : list
            A list of seed_T of top
        Note:
        ------------------
            coming soon
        '''
        seed_T = {}
        if model is None:
            print('a model need to be given.')
            return
        else:
            temp_model = model.copy()
            temp_model.update_seed_T([],override=True)
            res = temp_model.diffusion()
        
        node_ctr = {}
        for node in nx.get_node_attributes(res[0],'active_time').items():
            ctr = 0 # contribution
            for nbr in nx.neighbors(res[0]):
                if nbr in nx.get_node_attributes(res[0],'active_time'):
                    ctr+=1
            node_ctr[node]=ctr

        if k>len(node_ctr):
            print(f'cannot find k truth nodes, only {len(node_ctr)} nodes found.')
            return node_ctr    
        else:
            while(k):
                max_ctr = 0
                for node in node_ctr.items():
                    if node[1]>max_ctr:
                        chosen_node = node[0]
                        max_ctr = node[1]
                seed_T[chosen_node] = max_ctr
                node_ctr.pop(chosen_node)
                k-=1
        
        return seed_T

# %%

'''------------------ test below -----------------------'''
G = nx.Graph()
G.add_edges_from([(1,2),(1,7),(2,3),(3,4),(3,5),(4,6),(7,6),(7,8),(8,9),(9,5)])
nx.draw_networkx(G,pos=nx.circular_layout(G),with_labels=True,font_color='white')
#%%
model = LTD1DT.model_V2(G,False,[],[6,8])
G2 = model.get_initialized_G()
nx.get_node_attributes(G2,'status')
# %%
nx.get_node_attributes(G2,'i_threshold')
#%%
nx.get_node_attributes(G2,'d_threshold')
# %%
result = model.diffusion()
#%%
un_al = unconstrained_algorithm()
un_al.pagerank(G,)
# %%