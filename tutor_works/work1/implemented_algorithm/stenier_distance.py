#%%
import queue
import numpy as np
import networkx as nx
#%%
#%%
def spfa(dp:np.ndarray,q:queue.Queue,s:int,G:nx.Graph):
    visited  = np.zeros(nx.number_of_nodes(G)+1)

    while q.empty()!=True:
        q_top = q.get()

        if visited[q_top[1]]:
            continue

        visited[q_top[1]]=1

        for node in nx.neighbors(G,q_top[1]):
            if dp[node,s]>dp[q_top[1],s]+1:
                dp[node,s]=dp[q_top[1],s]+1
                temp = []
                temp.append(dp[node,s])
                temp.append(node)
                q.put(temp)


#%%
def mini_steiner_tree(G:nx.Graph,node_set:list):
    key_point_num = len(node_set)
    
    # create dp matrix (the G's node label from 1 to n, so the dp[0] wasn's used)
    dp = np.full((nx.number_of_nodes(G)+1,(1<<key_point_num)+1),fill_value=np.inf)
    q = queue.Queue()# For spfa
    # init dp[node_set[i],1<<(i-1)]=0
    for i in range(1,key_point_num+1):
        dp[node_set[i-1],1<<(i-1)]=0


    for s in range(1,(1<<key_point_num)):
        for i in range(1,nx.number_of_nodes(G)+1):
            # divide node_set into sub1 and sub2
            subs = s&(s-1)
            while(subs):
                dp[i][s]=min(dp[i][s],dp[i][subs]+dp[i][s^subs])
                subs = s&(subs-1)
            
            if dp[i][s]!=np.inf:
                temp = []
                temp.append(dp[i][s])
                temp.append(i)
                q.put(temp)
        spfa(dp,q,s,G)

    return dp[node_set[1],(1<<key_point_num)-1]

