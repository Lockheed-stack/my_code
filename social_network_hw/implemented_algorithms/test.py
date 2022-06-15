#%%
import torch
from Adamic_Adar import Adamic_Adar
from CommonNeighbors import common_neighbors
from JaccardCoefficient import Jaccard

from LocalRandomWalk import score_LRW
from InfluenceBasedLocalRandomWalk import score_ILRW
from SuperposeRandomWalk import score_SRW
from InfluenceBasedSuperposeRandomWalk import score_ISRW
import evaluate

import numpy as np
import networkx as nx
from sklearn.metrics import roc_curve,auc
# %%
G_karate = nx.read_adjlist('../soc-karate/soc-karate.mtx',nodetype=int)
G_dolphins = nx.read_adjlist('../dolphins/dolphins.mtx',nodetype=int)
G_netScience = nx.read_adjlist('../netscience/trimed_netScience.csv',nodetype=int,delimiter=',')
#%%
G_karate_train,G_karate_sample_edges = evaluate.random_remove_edges(G_karate,0.15)
G_dolphins_train,G_dolphins_sample_edges = evaluate.random_remove_edges(G_dolphins,0.15)
G_netScience_train,G_netScience_sample_edges = evaluate.random_remove_edges(G_netScience,0.15)
# %%
karate_cn_score = common_neighbors(G_karate_train,34)
dolphins_cn_score = common_neighbors(G_dolphins_train,18)
sci_cn_score = common_neighbors(G_netScience_train,34)
#print(cn_score)
karate_cn_auc=evaluate.AUC(karate_cn_score,G_karate,34)
dolphins_cn_auc = evaluate.AUC(dolphins_cn_score,G_dolphins,18)
sci_cn_auc = evaluate.AUC(sci_cn_score,G_netScience,34)
print(f"karate_cn_auc:{karate_cn_auc}\n dolphins_cn_score:{dolphins_cn_auc}\n sci_cn_score:{sci_cn_auc}")
print(evaluate.precision(10,karate_cn_score,G_karate,34))
print(evaluate.precision(10,dolphins_cn_score,G_karate,18))
print(evaluate.precision(100,sci_cn_score,G_karate,34))
# %%
karate_JC_score = Jaccard(G_karate_train,34)
dolphins_JC_score = Jaccard(G_dolphins_train,18)
sci_JC_score = Jaccard(G_netScience_train,34)
#print(JC_score)
karate_JC_auc=evaluate.AUC(karate_JC_score,G_karate,34)
dolphins_JC_auc = evaluate.AUC(dolphins_JC_score,G_dolphins,18)
sci_JC_auc = evaluate.AUC(sci_JC_score,G_netScience,34)
print(f"karate_Jc_auc:{karate_JC_auc}\n dolphins_JC_score:{dolphins_JC_auc}\n sci_JC_score:{sci_JC_auc}")
print(evaluate.precision(10,karate_JC_score,G_karate,34))
print(evaluate.precision(15,dolphins_JC_score,G_karate,18))
print(evaluate.precision(100,sci_JC_score,G_karate,34))
# %%
karate_AA_score = Adamic_Adar(G_karate_train,34)
dolphins_AA_score = Adamic_Adar(G_dolphins_train,18)
sci_AA_score = Adamic_Adar(G_netScience_train,34)
#print(AA_score)
karate_AA_auc=evaluate.AUC(karate_AA_score,G_karate,34)
dolphins_AA_auc = evaluate.AUC(dolphins_AA_score,G_dolphins,18)
sci_AA_auc = evaluate.AUC(sci_AA_score,G_netScience,34)
print(f"karate_AA_auc:{karate_AA_auc}\n dolphins_AA_score:{dolphins_AA_auc}\n sci_AA_score:{sci_AA_auc}")
print(evaluate.precision(10,karate_AA_score,G_karate,34))
print(evaluate.precision(15,dolphins_AA_score,G_karate,18))
print(evaluate.precision(100,sci_AA_score,G_karate,34))
#%%
karate_lrw_score = score_LRW(G_karate_train,3,34)
dolphins_lrw_score = score_LRW(G_dolphins_train,5,18)
sci_lrw_score = score_LRW(G_netScience_train,2,34)

karate_lrw_auc = evaluate.AUC(karate_lrw_score,G_karate,34)
dolphins_lrw_auc = evaluate.AUC(dolphins_lrw_score,G_dolphins,18)
sci_lrw_auc = evaluate.AUC(sci_lrw_score,G_netScience,34)
print(f"karate_lrw_auc:{karate_lrw_auc}\n dolphins_lrw_score:{dolphins_lrw_auc}\n sci_lrw_score:{sci_lrw_auc}")
print(evaluate.precision(10,karate_lrw_score,G_karate,34))
print(evaluate.precision(15,dolphins_lrw_score,G_karate,18))
print(evaluate.precision(100,sci_lrw_score,G_karate,34))
# %%
karate_srw_score = score_SRW(G_karate_train,2,34)
dolphins_srw_score = score_SRW(G_dolphins_train,5,18)
sci_srw_score = score_SRW(G_netScience_train,3,34)

karate_srw_auc = evaluate.AUC(karate_srw_score,G_karate,34)
dolphins_srw_auc = evaluate.AUC(dolphins_srw_score,G_dolphins,18)
sci_srw_auc = evaluate.AUC(sci_srw_score,G_netScience,34)
print(f"karate_srw_auc:{karate_srw_auc}\n dolphins_srw_score:{dolphins_srw_auc}\n sci_srw_score:{sci_srw_auc}")
print(evaluate.precision(10,karate_srw_score,G_karate,34))
print(evaluate.precision(20,dolphins_srw_score,G_karate,18))
print(evaluate.precision(100,sci_srw_score,G_karate,34))
# %%
s = [G_dolphins_train.subgraph(c) for c in  nx.connected_components(G_dolphins_train)]
# %%
nx.radius(s[0])
# %%
G_karate_sample_edges
# %%
cn_score_karate=common_neighbors(G_karate_train,34)
scores = []
y_true=[]
for node in cn_score_karate:
    if node[0] in nx.neighbors(G_karate,34):
        y_true.append(1)
    else:
        y_true.append(0)
    scores.append(node[1])
fpr,tpr,thresholds=roc_curve(y_true,scores,)
fpr
#%%
thresholds
# %%
all_node_auc_karate=[]
already_test_nodes={}
for node in G_dolphins_sample_edges:
    if node[0] not in already_test_nodes:
        already_test_nodes[node[0]]=1
        all_node_auc_karate.append(evaluate.AUC(Jaccard(G_dolphins_train,node[0]),G_dolphins,node[0]))
    if node[1] not in already_test_nodes:
        already_test_nodes[node[1]]=1
        all_node_auc_karate.append(evaluate.AUC(Jaccard(G_dolphins_train,node[1]),G_dolphins,node[1]))
    print(np.mean(all_node_auc_karate))
# %%
all_node_auc_karate=[]
already_test_nodes={}
for node in G_dolphins_sample_edges:
    if node[0] not in already_test_nodes:
        already_test_nodes[node[0]]=1
        all_node_auc_karate.append(evaluate.AUC(Adamic_Adar(G_dolphins_train,node[0]),G_dolphins,node[0]))
    if node[1] not in already_test_nodes:
        already_test_nodes[node[1]]=1
        all_node_auc_karate.append(evaluate.AUC(Adamic_Adar(G_dolphins_train,node[1]),G_dolphins,node[1]))
    print(np.mean(all_node_auc_karate))
# %%
all_node_auc_karate=[]
already_test_nodes={}
for node in G_dolphins_sample_edges:
    if node[0] not in already_test_nodes:
        already_test_nodes[node[0]]=1
        all_node_auc_karate.append(evaluate.AUC(score_LRW(G_dolphins_train,4,node[0]),G_dolphins,node[0]))
    if node[1] not in already_test_nodes:
        already_test_nodes[node[1]]=1
        all_node_auc_karate.append(evaluate.AUC(score_LRW(G_dolphins_train,4,node[1]),G_dolphins,node[1]))
    print(np.mean(all_node_auc_karate))
# %%
all_node_auc_karate=[]
already_test_nodes={}
for node in G_dolphins_sample_edges:
    if node[0] not in already_test_nodes:
        already_test_nodes[node[0]]=1
        all_node_auc_karate.append(evaluate.AUC(score_ILRW(G_dolphins_train,4,node[0]),G_dolphins,node[0]))
    if node[1] not in already_test_nodes:
        already_test_nodes[node[1]]=1
        all_node_auc_karate.append(evaluate.AUC(score_ILRW(G_dolphins_train,4,node[1]),G_dolphins,node[1]))
np.mean(all_node_auc_karate) 
# %%
all_node_auc_karate=[]
already_test_nodes={}
for node in G_dolphins_sample_edges:
    if node[0] not in already_test_nodes:
        already_test_nodes[node[0]]=1
        all_node_auc_karate.append(evaluate.AUC(score_SRW(G_dolphins_train,5,node[0]),G_dolphins,node[0]))
    if node[1] not in already_test_nodes:
        already_test_nodes[node[1]]=1
        all_node_auc_karate.append(evaluate.AUC(score_SRW(G_dolphins_train,5,node[1]),G_dolphins,node[1]))
np.mean(all_node_auc_karate) 
# %%
all_node_auc_karate=[]
already_test_nodes={}
for node in G_dolphins_sample_edges:
    if node[0] not in already_test_nodes:
        already_test_nodes[node[0]]=1
        all_node_auc_karate.append(evaluate.AUC(score_ISRW(G_dolphins_train,5,node[0]),G_dolphins,node[0]))
    if node[1] not in already_test_nodes:
        already_test_nodes[node[1]]=1
        all_node_auc_karate.append(evaluate.AUC(score_ISRW(G_dolphins_train,5,node[1]),G_dolphins,node[1]))
np.mean(all_node_auc_karate) 
# %%
G = nx.read_adjlist('../SCHOLAT Link Prediction/train.csv',nodetype=int,delimiter=',')
# %%
output = score_LRW(G,5,354)
output
# %%
# %%
int(0.1*len(sci_cn_score))
# %%
assert True is True
# %%
cn_score_karate=common_neighbors(G_karate_train,34)
scores = []
y_true=[]
for node in cn_score_karate:
    if node[0] in nx.neighbors(G_karate,34):
        y_true.append(1)
    else:
        y_true.append(0)
    scores.append(node[1])
fpr,tpr,thresholds=roc_curve(y_true,scores,)
#%%
import matplotlib.pyplot as plt 
# %%
fig,ax = plt.subplots()
lw=2
ax.plot(fpr,
    tpr,
    color="darkorange",
    label="ROC curve")
ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()
# %%
already_test_nodes={}
scores = []
y_true=[]
for node in G_karate_sample_edges:
    if node[0] not in already_test_nodes:
        already_test_nodes[node[0]]=1
        cn_score = common_neighbors(G_karate_train,node[0])
        for node_y in cn_score:
            if node_y[0] in nx.neighbors(G_karate,node[0]):
                y_true.append(1)
            else:
                y_true.append(0)
            scores.append(node_y[1])
    if node[1] not in already_test_nodes:
        already_test_nodes[node[1]]=1
        cn_score = common_neighbors(G_karate_train,node[1])
        for node_y in cn_score:
            if node_y[0] in nx.neighbors(G_karate,node[1]):
                y_true.append(1)
            else:
                y_true.append(0)
            scores.append(node_y[1])
fpr,tpr,thresholds=roc_curve(y_true,scores,)
# %%
y_true
# %%
fig,ax = plt.subplots()
lw=2
ax.plot(fpr,
    tpr,
    color="darkorange",
    label="ROC curve")
ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()
# %%
def roc_calc(G_train,G_test,sample_edges,score_callback_func,Is_rw=False,time_step=None):
    already_test_nodes={}
    scores = []
    y_true=[]
    if Is_rw==False:
        for node in sample_edges:
            if node[0] not in already_test_nodes:
                already_test_nodes[node[0]]=1
                score_res = score_callback_func(G_train,node[0])
                for node_y in score_res:
                    if node_y[0] in nx.neighbors(G_test,node[0]):
                        y_true.append(1)
                    else:
                        y_true.append(0)
                    scores.append(node_y[1])
            if node[1] not in already_test_nodes:
                already_test_nodes[node[1]]=1
                score_res = score_callback_func(G_train,node[1])
                for node_y in score_res:
                    if node_y[0] in nx.neighbors(G_test,node[1]):
                        y_true.append(1)
                    else:
                        y_true.append(0)
                    scores.append(node_y[1])
    else:
        tensor_scores = []
        for node in sample_edges:
            if node[0] not in already_test_nodes:
                already_test_nodes[node[0]]=1
                score_res = score_callback_func(G_train,time_step,node[0])
                for node_y in score_res:
                    if node_y[0] in nx.neighbors(G_test,node[0]):
                        y_true.append(1)
                    else:
                        y_true.append(0)
                    tensor_scores.append(node_y[1])
            if node[1] not in already_test_nodes:
                already_test_nodes[node[1]]=1
                score_res = score_callback_func(G_train,time_step,node[1])
                for node_y in score_res:
                    if node_y[0] in nx.neighbors(G_test,node[1]):
                        y_true.append(1)
                    else:
                        y_true.append(0)
                    tensor_scores.append(node_y[1])
        for t in tensor_scores:
            scores.append(torch.Tensor.cpu(t))
            
    return y_true,scores
#%%
y_true,scores = roc_calc(G_dolphins_train,G_dolphins,G_dolphins_sample_edges,
                         Jaccard,)
# %%
fpr,tpr,thresholds=roc_curve(y_true,scores,)
# %%
fig,ax = plt.subplots()
lw=2
ax.plot(fpr,
    tpr,
    color="darkorange",
    label="ROC curve: dolphins")
ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()
# %%
cn_karate_y_true,cn_karate_scores = roc_calc(G_karate_train,G_karate,G_karate_sample_edges
                                       ,common_neighbors)
jc_karate_y_true,jc_karate_scores = roc_calc(G_karate_train,G_karate,G_karate_sample_edges
                                       ,Jaccard)
aa_karate_y_true,aa_karate_scores = roc_calc(G_karate_train,G_karate,G_karate_sample_edges
                                       ,Adamic_Adar)
cn_fpr,cn_tpr,cn_thresholds = roc_curve(cn_karate_y_true,cn_karate_scores)
jc_fpr,jc_tpr,jc_thresholds = roc_curve(jc_karate_y_true,jc_karate_scores)
aa_fpr,aa_tpr,jc_thresholds = roc_curve(aa_karate_y_true,aa_karate_scores)
#%%
lrw_karate_y_true,lrw_karate_scores = roc_calc(G_karate_train,G_karate,G_karate_sample_edges,
                                               score_LRW,True,3)
lrw_fpr,lrw_tpr,lrw_thresholds = roc_curve(lrw_karate_y_true,lrw_karate_scores)
#%%
srw_karate_y_true,srw_karate_scores = roc_calc(G_karate_train,G_karate,G_karate_sample_edges,
                                               score_SRW,True,3)
srw_fpr,srw_tpr,srw_th = roc_curve(srw_karate_y_true,srw_karate_scores)
#%%
ilrw_karate_y_true,ilrw_karate_scores = roc_calc(G_karate_train,G_karate,G_karate_sample_edges,
                                               score_ILRW,True,3)
ilrw_fpr,ilrw_tpr,ilrw_thresholds = roc_curve(ilrw_karate_y_true,ilrw_karate_scores)
# %%
isrw_karate_y_true,isrw_karate_scores = roc_calc(G_karate_train,G_karate,G_karate_sample_edges,
                                               score_ISRW,True,3)
isrw_fpr,isrw_tpr,isrw_th = roc_curve(isrw_karate_y_true,isrw_karate_scores)
#%%
fig,ax = plt.subplots()
ax.plot(cn_fpr,cn_tpr,label='CN,area=%0.3f'%auc(cn_fpr,cn_tpr))
ax.plot(jc_fpr,jc_tpr,label='JC,area=%0.3f'%auc(jc_fpr,jc_tpr))
ax.plot(aa_fpr,aa_tpr,label='AA,area=%0.3f'%auc(aa_fpr,aa_tpr))
ax.plot(lrw_fpr,lrw_tpr,label='LRW,area=%0.3f'%auc(lrw_fpr,lrw_tpr))
ax.plot(srw_fpr,srw_tpr,label='SRW,area=%0.3f'%auc(srw_fpr,srw_tpr))
ax.plot(ilrw_fpr,ilrw_tpr,label='LRW,area=%0.3f'%auc(ilrw_fpr,ilrw_tpr))
ax.plot(isrw_fpr,isrw_tpr,label='SRW,area=%0.3f'%auc(isrw_fpr,isrw_tpr))
ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Karate ROC")
plt.legend(loc="lower right")
plt.savefig('../outpic/karate_roc.jpg',dpi=300,bbox_inches='tight')
plt.show()
# %%
mem_score = []
for t in lrw_karate_scores:
    mem_score.append(t.cpu())
mem_score
# %%
