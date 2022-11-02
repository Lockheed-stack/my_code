#%%
import networkx as nx
import matplotlib.pyplot as plt
from model import *
import numpy as np
# %%
G1 = nx.read_adjlist('../dataset/dolphins.mtx',nodetype=int)
G2 = nx.read_adjlist('../dataset/trimed_netScience.csv',nodetype=int,delimiter=',')
G3 = nx.read_adjlist('../dataset/fb-pages-food.txt',nodetype=int,delimiter=',')
G4 = nx.read_adjlist('../dataset/USpowerGrid.mtx',nodetype=int,)
# %%
model1 = model(G1,0.004,)
model1.authoritative_T
# %%
model2 = model(G2,0.004)
model2.authoritative_T
# %%
model3 = model(G3,0.004)
model3.authoritative_T
# %%
model4 = model(G4,0.004)
model4.authoritative_T
# %%
avg_detected_time1=[]
for i in np.linspace(0.01,0.05,9):
    detected_time = 0
    for j in range(100):
        model1.refresh_i_c_threshold()
        r_n = model1.generate_R_nodes(0,i,)
        detected_time += model1.before_detected_diffusion(r_n,)[1]
    avg_detected_time1.append(detected_time/100)
avg_detected_time1
#%%
avg_detected_time2=[]
for i in np.linspace(0.01,0.05,9):
    detected_time = 0
    for j in range(100):
        model2.refresh_i_c_threshold()
        r_n = model2.generate_R_nodes(0,i,)
        detected_time += model2.before_detected_diffusion(r_n,)[1]
    avg_detected_time2.append(detected_time/100)
avg_detected_time2
# %%
avg_detected_time3=[]
for i in np.linspace(0.01,0.05,9):
    detected_time = 0
    for j in range(100):
        model3.refresh_i_c_threshold()
        r_n = model3.generate_R_nodes(0,i,)
        detected_time += model3.before_detected_diffusion(r_n,)[1]
    avg_detected_time3.append(detected_time/100)
avg_detected_time3
# %%
avg_detected_time4=[]
for i in np.linspace(0.01,0.05,9):
    detected_time = 0
    for j in range(100):
        model4.refresh_i_c_threshold()
        r_n = model4.generate_R_nodes(0,i,)
        detected_time += model4.before_detected_diffusion(r_n,)[1]
    avg_detected_time4.append(detected_time/100)
avg_detected_time4
# %%
fig,axs = plt.subplots(1,1,figsize=[8,5],facecolor='w')
x = np.linspace(0.01,0.05,9)
axs.plot(x,avg_detected_time1,label='dolphin',marker='o')
axs.plot(x,avg_detected_time2,label='netScience',marker='x')
axs.plot(x,avg_detected_time3,label='food',marker='*')
axs.plot(x,avg_detected_time4,label='USpower',marker='^')
axs.set_xlabel('Rumor nodes percent')
axs.set_ylabel('average first detect time')
plt.legend()
plt.grid(True)
plt.savefig('../res_pic/first_detect_time.tif',dpi=300)
plt.show()
# %%
np.linspace(0.01,0.05,9)
# %%
