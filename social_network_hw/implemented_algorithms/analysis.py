#%%
from matplotlib import markers
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import evaluate

from Adamic_Adar import Adamic_Adar
from CommonNeighbors import common_neighbors
from JaccardCoefficient import Jaccard

from LocalRandomWalk import score_LRW
from InfluenceBasedLocalRandomWalk import score_ILRW
from SuperposeRandomWalk import score_SRW
from InfluenceBasedSuperposeRandomWalk import score_ISRW
#%%
G_karate = nx.read_adjlist('../soc-karate/soc-karate.mtx',nodetype=int)
G_dolphins = nx.read_adjlist('../dolphins/dolphins.mtx',nodetype=int)
G_netScience = nx.read_adjlist('../netscience/trimed_netScience.csv',nodetype=int,delimiter=',')
G_scholat_train = nx.read_adjlist('../SCHOLAT Link Prediction/train.csv',nodetype=int,delimiter=',')
G_scholat_test = nx.read_adjlist('../SCHOLAT Link Prediction/test.csv',nodetype=int,delimiter=',')
#%%
G_karate_train,G_karate_sample_edges = evaluate.random_remove_edges(G_karate,0.15)
G_dolphins_train,G_dolphins_sample_edges = evaluate.random_remove_edges(G_dolphins,0.15)
G_netScience_train,G_netScience_sample_edges = evaluate.random_remove_edges(G_netScience,0.15)
#%%
def start_analysis(G_sample_edge=None,G_train=None,G_test=None,sample_rate=1.0,exec_srw=False):
    if G_sample_edge is not None:
        all_node_auc_CN=[]
        all_node_auc_JC=[]
        all_node_auc_AA=[]
        all_node_auc_lrw=[]
        all_node_auc_ilrw=[]
        all_node_auc_srw=[]
        all_node_auc_isrw=[]
        time_steps = step_decision(G_train)
        
        already_test_nodes={}
        if sample_rate==1.0:
            sample_rate_edges=G_sample_edge
        else:
            sample_rate_edges=pd.DataFrame(G_sample_edge).sample(frac=sample_rate,replace=False).values
        
        for node in sample_rate_edges:
            if node[0] not in already_test_nodes:
                already_test_nodes[node[0]] = 1
                all_node_auc_CN.append(evaluate.AUC(common_neighbors(G_train,node[0]),G_test,node[0]))
                all_node_auc_JC.append(evaluate.AUC(Jaccard(G_train,node[0]),G_test,node[0]))
                all_node_auc_AA.append(evaluate.AUC(Adamic_Adar(G_train,node[0]),G_test,node[0]))
                all_node_auc_lrw.append(evaluate.AUC(score_LRW(G_train,time_steps,node[0]),G_test,node[0]))
                all_node_auc_ilrw.append(evaluate.AUC(score_ILRW(G_train,time_steps,node[0]),G_test,node[0]))
                if exec_srw is True:
                    all_node_auc_srw.append(evaluate.AUC(score_SRW(G_train,time_steps,node[0]),G_test,node[0]))
                    all_node_auc_isrw.append(evaluate.AUC(score_ISRW(G_train,time_steps,node[0]),G_test,node[0]))
            if node[1] not in already_test_nodes:
                already_test_nodes[node[1]]=1
                all_node_auc_CN.append(evaluate.AUC(common_neighbors(G_train,node[1]),G_test,node[1]))
                all_node_auc_JC.append(evaluate.AUC(Jaccard(G_train,node[1]),G_test,node[1]))
                all_node_auc_AA.append(evaluate.AUC(Adamic_Adar(G_train,node[1]),G_test,node[1]))
                all_node_auc_lrw.append(evaluate.AUC(score_LRW(G_train,time_steps,node[1]),G_test,node[1]))
                all_node_auc_ilrw.append(evaluate.AUC(score_ILRW(G_train,time_steps,node[1]),G_test,node[1]))
                if exec_srw is True:
                    all_node_auc_srw.append(evaluate.AUC(score_SRW(G_train,time_steps,node[1]),G_test,node[1]))
                    all_node_auc_isrw.append(evaluate.AUC(score_ISRW(G_train,time_steps,node[1]),G_test,node[1]))
                
    elif G_test is not None:
        pass
    
    result_gather = {}
    result_gather['CN']=all_node_auc_CN
    result_gather['JC']=all_node_auc_JC
    result_gather['AA']=all_node_auc_AA
    result_gather['LRW']=all_node_auc_lrw
    result_gather['ILRW']=all_node_auc_ilrw
    if exec_srw is True:
        result_gather['SRW']=all_node_auc_srw
        result_gather['ISRW']=all_node_auc_isrw
    
    return result_gather

#%%
def difference_step_RW(G_train,G_test,sample_edges,sample_rate=1.0,exec_srw = False):
    '''
    @return a df that show the effect of different step sizes on results
    '''
    if exec_srw is True:
        df_column = ['LRW','ILRW','SRW','ISRW']
    else:
        df_column = ['LRW','ILRW']

    # df_column = ['LRW','ILRW','SRW','ISRW']
    df = pd.DataFrame(columns=df_column,index=[1,2,3,4,5,6],
                      data=0,dtype=float)
    # node = pd.DataFrame(sample_edges).sample(1,replace=False).values[0][0]
    # for time_steps in range(1,7):
    #     auc_lrw = evaluate.AUC(score_LRW(G_train,time_steps,node),G_test,node)
    #     auc_ilrw = evaluate.AUC(score_ILRW(G_train,time_steps,node),G_test,node)
    #     if exec_srw is True:
    #         auc_srw = evaluate.AUC(score_SRW(G_train,time_steps,node),G_test,node)
    #         auc_isrw = evaluate.AUC(score_ISRW(G_train,time_steps,node),G_test,node)
    #     df_data = [auc_lrw,auc_ilrw,auc_srw,auc_isrw]
    #     df.loc[time_steps]=df_data
    if sample_rate==1.0:
        sample_rate_edges = sample_edges
    else:
        sample_rate_edges = pd.DataFrame(sample_edges).sample(frac=sample_rate,replace=False).values
    
    for time_steps in range(1,7):
        already_test_nodes={}
        df_data=[]
        if exec_srw is True:
            all_node_auc_lrw=[]
            all_node_auc_ilrw=[]
            all_node_auc_srw=[]
            all_node_auc_isrw=[]
        else:
            all_node_auc_lrw=[]
            all_node_auc_ilrw=[]
            
        for node in sample_rate_edges:
            if node[0] not in already_test_nodes:
                already_test_nodes[node[0]] = 1
                all_node_auc_lrw.append(evaluate.AUC(score_LRW(G_train,time_steps,node[0]),G_test,node[0]))
                all_node_auc_ilrw.append(evaluate.AUC(score_ILRW(G_train,time_steps,node[0]),G_test,node[0]))
                if exec_srw is True:
                    all_node_auc_srw.append(evaluate.AUC(score_SRW(G_train,time_steps,node[0]),G_test,node[0]))
                    all_node_auc_isrw.append(evaluate.AUC(score_ISRW(G_train,time_steps,node[0]),G_test,node[0]))
            if node[1] not in already_test_nodes:
                already_test_nodes[node[1]]=1
                all_node_auc_lrw.append(evaluate.AUC(score_LRW(G_train,time_steps,node[1]),G_test,node[1]))
                all_node_auc_ilrw.append(evaluate.AUC(score_ILRW(G_train,time_steps,node[1]),G_test,node[1]))
                if exec_srw is True:
                    all_node_auc_srw.append(evaluate.AUC(score_SRW(G_train,time_steps,node[1]),G_test,node[1]))
                    all_node_auc_isrw.append(evaluate.AUC(score_ISRW(G_train,time_steps,node[1]),G_test,node[1]))
        
        df_data.append(np.mean(all_node_auc_lrw))
        df_data.append(np.mean(all_node_auc_isrw))
        if exec_srw is True:
            df_data.append(np.mean(all_node_auc_srw))
            df_data.append(np.mean(all_node_auc_isrw))
            
        df.loc[time_steps]=df_data
    
    return df
#%%
def step_decision(G_train):
    largest_cc = max(nx.connected_components(G_train), key=len)
    
    return nx.radius(G_train.subgraph(largest_cc))

#%%
def pandas_process(Result_gather):
    '''
    @return a observed fraction Dataframe
    @Result_gather: dict , get from 'start analysis' function
    '''
    df = pd.DataFrame(Result_gather)
    observed_frac_df = pd.DataFrame(columns=df.columns,
                                    index=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                                    data=0,dtype=float)
    
    for i in np.arange(0.1,1.1,step=0.1):
        df_index = int(len(df.index)*i)
        observed_frac_df.iloc[int(i*10)-1]=df.iloc[:df_index].mean(axis=0).values


    return observed_frac_df

#%%
#------------------------ test below ------------------------------#
#%%
dolphin_diff_step_rw = difference_step_RW(G_dolphins_train,G_dolphins,G_dolphins_sample_edges,exec_srw=True)
dolphin_diff_step_rw
# %%
dolphin_radius =nx.radius(G_dolphins_train.subgraph(max(nx.connected_components(G_dolphins_train),key=len)))
dolphin_radius
# %%
karate_radius = nx.radius(G_karate_train.subgraph(max(nx.connected_components(G_karate_train),key=len)))
karate_radius
# %%
fig,ax1 =plt.subplots()
ax1.plot(sci_diff_step_rw,marker='8')
ax1.legend(sci_diff_step_rw.columns)
ax1.set_xlabel('Random Walk Steps')
ax1.set_ylabel('AUC')
ax1.set_title("netScience")
plt.savefig('../outpic/diff_steps_netSci_auc.jpg',dpi=400,bbox_inches='tight')
plt.show()
# %%
sci_diff_step_rw = difference_step_RW(G_netScience_train,G_netScience,G_netScience_sample_edges,0.05,True)
sci_diff_step_rw
# %%
len(pd.DataFrame(G_netScience_sample_edges).sample(frac=0.05,replace=False).values)
# %%
G_dolphins_sample_edges
# %%
ax1=sci_diff_step_rw.plot(kind='bar',title='netScience',rot=1,
                          legend=True,xlabel='Random Walk Steps',ylabel='AUC')
plt.savefig('../outpic/diff_steps_netScience_bar_auc.jpg',dpi=400,bbox_inches='tight')
plt.show()
# %%
len(G_scholat_test.edges)
# %%
pd.DataFrame(G_scholat_test.edges)
# %%
nx.radius(G_karate)
# %%
karate_diff_step_rw = difference_step_RW(G_karate_train,G_karate,G_karate_sample_edges,exec_srw=True)
karate_diff_step_rw
# %%
# %%
netsci_radius = nx.radius(G_netScience_train.subgraph(max(nx.connected_components(G_netScience_train),key=len)))
netsci_radius
# %%
a = []
b = {}
for node in dict(nx.shortest_path_length(G)).values():
    for v in node.values():
        a.append(v)
a
# %%
np.shape(a)
# %%
np.mean(a)
# %%
G = G_scholat_train.subgraph(max(nx.connected_components(G_scholat_train),key=len))
G.number_of_nodes()
# %%
G.number_of_edges()
# %%
nx.number_of_edges(G_netScience_train)
# %%
nx.degree(G_karate_train)
# %%
karate_df = pd.DataFrame(nx.degree(G_karate_train),columns=['node','degree'])
dolphins_df = pd.DataFrame(nx.degree(G_dolphins_train),columns=['node','degree'])
sci_df = pd.DataFrame(nx.degree(G_netScience_train),columns=['node','degree'])
scho_df = pd.DataFrame(nx.degree(G_scholat_train),columns=['node','degree'])
# %%
k_s = pd.value_counts(karate_df['degree'])
d_s = pd.value_counts(dolphins_df['degree']).drop(0)
sci_s = pd.value_counts(sci_df['degree']).drop(0)
scho_s = pd.value_counts(scho_df['degree'])
# %%
fig,axs = plt.subplots(2,3,figsize=(15,10))
axs[0][0].bar(k_s.index,k_s.values,color=['coral', 'dodgerblue'])
axs[0][0].set_xticks(np.arange(1,18,1))
axs[0][0].set_title('Karate')

axs[0][1].bar(d_s.index,d_s.values,color=['coral', 'dodgerblue'])
axs[0][1].set_title('Dolphins')

axs[0][2].bar(sci_s.index,sci_s.values,color=['coral', 'dodgerblue'])
axs[0][2].set_title('netScience')

axs[1][0] = plt.subplot(212)
axs[1][0].bar(scho_s.index,scho_s.values,color=['coral', 'dodgerblue'])
axs[1][0].set_title('Scholat')

for ax in axs:
    for i in ax:
        i.set_xlabel('degree')
        i.set_ylabel('counts')
        i.grid(True,alpha=0.6)
plt.savefig('../outpic/degree_distribution.jpg',dpi=1000,bbox_inches='tight')
plt.show()
# %%
max(nx.degree(G_scholat_train),key=lambda kv:kv[1])

# %%
for ax in axs:
    print(type(ax))
# %%
karate_auc = start_analysis(G_karate_sample_edges,G_karate_train,G_karate,exec_srw=True)
karate_auc_df = pandas_process(karate_auc)
karate_auc_df
# %%
fig,ax = plt.subplots()
ax.plot(karate_auc_df,marker='*')
ax.legend(karate_auc_df.columns)
ax.set_xticks(np.arange(0.1,1.1,0.1))
ax.set_xlabel('Observed fraction')
ax.set_ylabel('AUC')
ax.set_title('Karate')
plt.savefig('../outpic/karate_auc.jpg',dpi=300,bbox_inches='tight')
plt.show()
# %%
dolphins_auc = start_analysis(G_dolphins_sample_edges,G_dolphins_train,G_dolphins,exec_srw=True)
dolphins_auc_df = pandas_process(dolphins_auc)
dolphins_auc_df
# %%
fig,ax = plt.subplots()
ax.plot(dolphins_auc_df,marker='*')
ax.legend(dolphins_auc_df.columns)
ax.set_xticks(np.arange(0.1,1.1,0.1))
ax.set_xlabel('Observed fraction')
ax.set_ylabel('AUC')
ax.set_title('Dolphins')
plt.savefig('../outpic/dolphins_auc.jpg',dpi=300,bbox_inches='tight')
plt.show()
# %%
sci_auc = start_analysis(G_netScience_sample_edges,G_netScience_train,G_netScience,sample_rate=0.1,exec_srw=True)
dolphins_auc_df = pandas_process(dolphins_auc)
dolphins_auc_df
# %%
