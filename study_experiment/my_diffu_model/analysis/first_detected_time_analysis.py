#%%
import networkx as nx
import matplotlib.pyplot as plt
from model import *
import numpy as np
# %%
G1 = nx.read_adjlist('../dataset/dolphins.mtx',nodetype=int)
G2 = nx.read_adjlist('../dataset/fb-pages-food.txt',nodetype=int,delimiter=',')
G3 = nx.read_adjlist('../dataset/trimed_netScience.csv',nodetype=int,delimiter=',')
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
#----------------- fix authoritative T nodes -----------------
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
avg_detected_time1 = [1.79,	1.9,	1.9,	2.33,	1.97,	1.99,	2.01,	2.25,	2.07]
avg_detected_time2 = [4.86,	5.7,	5.69,	5.15,	5.51,	5.08,	5.41,	5.19,	4.52]
avg_detected_time3 = [4.24,	4.19,	4.45,	4.93,	3.82,	4.24,	3.91,	3.74,	3.59]
avg_detected_time4 = [4.52,	3.9,	2.51,	2.11,	1.97,	1.31,	1.58,	1.46,	1.27]
fig,axs = plt.subplots(1,1,figsize=[8,5],facecolor='w')
x = np.linspace(0.01,0.05,9)
axs.plot(x,avg_detected_time1,label='dolphin',marker='o')
axs.plot(x,avg_detected_time2,label='food',marker='x')
axs.plot(x,avg_detected_time3,label='netscience',marker='*')
axs.plot(x,avg_detected_time4,label='USpower',marker='^')
axs.set_xlabel('Rumor nodes percent')
axs.set_ylabel('average time to first detection')
plt.title('(fixed authoritative T)authoritative T percent: 0.004')
plt.legend()
plt.grid(True)
plt.savefig('../res_pic/first_detect_time_1.tif',dpi=300)
plt.show()

# %%
#-------- completely random -------------
avg_detected_time1=[]
for i in np.linspace(0.01,0.05,9):
    detected_time = 0
    for j in range(100):
        model1.refresh_i_c_threshold(True,0.004)
        r_n = model1.generate_R_nodes(0,i,)
        detected_time += model1.before_detected_diffusion(r_n,)[1]
    avg_detected_time1.append(detected_time/100)
avg_detected_time1
#%%
avg_detected_time2=[]
for i in np.linspace(0.01,0.05,9):
    detected_time = 0
    for j in range(100):
        model2.refresh_i_c_threshold(True,0.004)
        r_n = model2.generate_R_nodes(0,i,)
        detected_time += model2.before_detected_diffusion(r_n,)[1]
    avg_detected_time2.append(detected_time/100)
avg_detected_time2
# %%
avg_detected_time3=[]
for i in np.linspace(0.01,0.05,9):
    detected_time = 0
    for j in range(100):
        model3.refresh_i_c_threshold(True,0.004)
        r_n = model3.generate_R_nodes(0,i,)
        detected_time += model3.before_detected_diffusion(r_n,)[1]
    avg_detected_time3.append(detected_time/100)
avg_detected_time3
# %%
avg_detected_time4=[]
for i in np.linspace(0.01,0.05,9):
    detected_time = 0
    for j in range(100):
        model4.refresh_i_c_threshold(True,0.004)
        r_n = model4.generate_R_nodes(0,i,)
        detected_time += model4.before_detected_diffusion(r_n,)[1]
    avg_detected_time4.append(detected_time/100)
avg_detected_time4
# %%
avg_detected_time1 = [2.02,	2.24,	2.01,	2.31,	2.27,	2.43,	2.44,	2.32,	2.28]
avg_detected_time2 = [4.52,	4.57,	4.39,	3.89,	4.03,	3.66,	4.0,	3.95,	3.29]
avg_detected_time3 = [4.4,	4.43,	3.69,	4.23,	3.88,	3.81,	4.08,	3.51,	3.13]
avg_detected_time4 = [5.87,	4.22,	3.41,	3.13,	1.88,	1.9,	1.78,	1.43,	1.76]
fig,axs = plt.subplots(1,1,figsize=[8,5],facecolor='w')
x = np.linspace(0.01,0.05,9)
axs.plot(x,avg_detected_time1,label='dolphin',marker='o')
axs.plot(x,avg_detected_time2,label='food',marker='x')
axs.plot(x,avg_detected_time3,label='netscience',marker='*')
axs.plot(x,avg_detected_time4,label='USpower',marker='^')
axs.set_xlabel('Rumor nodes percent')
axs.set_ylabel('average time to first detection')
plt.legend()
plt.grid(True)
plt.title('(completely random)authoritative T percent: 0.004')
plt.savefig('../res_pic/first_detect_time_2.tif',dpi=300)
plt.show()
# %%
np.linspace(0.001,0.01,num=10)
# %%
#---------------- different authoritative T percent -------------------
avg_detected_time1=[]
for i in np.linspace(0.001,0.01,10):
    detected_time = 0
    for j in range(100):
        model1.refresh_i_c_threshold(True,i)
        r_n = model1.generate_R_nodes(0,0.05)
        detected_time += model1.before_detected_diffusion(r_n,)[1]
    avg_detected_time1.append(detected_time/100)
avg_detected_time1
# %%
avg_detected_time2=[]
for i in np.linspace(0.001,0.01,10):
    detected_time = 0
    for j in range(100):
        model2.refresh_i_c_threshold(True,i)
        r_n = model2.generate_R_nodes(0,0.05)
        detected_time += model2.before_detected_diffusion(r_n,)[1]
    avg_detected_time2.append(detected_time/100)
avg_detected_time2
# %%
avg_detected_time3=[]
for i in np.linspace(0.001,0.01,10):
    detected_time = 0
    for j in range(100):
        model3.refresh_i_c_threshold(True,i)
        r_n = model3.generate_R_nodes(0,0.05)
        detected_time += model3.before_detected_diffusion(r_n,)[1]
    avg_detected_time3.append(detected_time/100)
avg_detected_time3
# %%
avg_detected_time4=[]
for i in np.linspace(0.001,0.01,10):
    detected_time = 0
    for j in range(100):
        model4.refresh_i_c_threshold(True,i)
        r_n = model4.generate_R_nodes(0,0.05)
        detected_time += model4.before_detected_diffusion(r_n,)[1]
    avg_detected_time4.append(detected_time/100)
avg_detected_time4
# %%
fig,axs = plt.subplots(1,1,figsize=[8,5],facecolor='w')
x = np.linspace(0.001,0.01,10)
axs.plot(x,avg_detected_time1,label='dolphin',marker='o')
axs.plot(x,avg_detected_time2,label='food',marker='x')
axs.plot(x,avg_detected_time3,label='netscience',marker='*')
axs.plot(x,avg_detected_time4,label='USpower',marker='^')
axs.set_xlabel('authoritative T nodes percent')
axs.set_ylabel('average time to first detection')
plt.legend()
plt.grid(True)
plt.title('Rumor nodes percent: 0.05')
plt.savefig('../res_pic/first_detect_time_3.tif',dpi=300)
plt.show()
# %%


#-------------- residual available nodes --------------------
avg_avail_nodes_1 = []
all_nodes = nx.number_of_nodes(G1)
for i in np.linspace(0.01,0.05,9):
    avail = 0
    for j in range(500):
        model1.refresh_i_c_threshold()
        r_n = model1.generate_R_nodes(0,i)
        result = model1.before_detected_diffusion(r_n,)
        avail += all_nodes - result[4][result[1]] - len(model1.authoritative_T)
    avg_avail_nodes_1.append(avail/500)
avg_avail_nodes_1
# %%
avg_avail_nodes_2 = []
all_nodes = nx.number_of_nodes(G2)
for i in np.linspace(0.01,0.05,9):
    avail = 0
    for j in range(500):
        model2.refresh_i_c_threshold()
        r_n = model2.generate_R_nodes(0,i)
        result = model2.before_detected_diffusion(r_n,)
        avail += all_nodes - result[4][result[1]] - len(model2.authoritative_T)
    avg_avail_nodes_2.append(avail/500)
avg_avail_nodes_2
# %%
avg_avail_nodes_3 = []
all_nodes = nx.number_of_nodes(G3)
for i in np.linspace(0.01,0.05,9):
    avail = 0
    for j in range(500):
        model3.refresh_i_c_threshold()
        r_n = model3.generate_R_nodes(0,i)
        result = model3.before_detected_diffusion(r_n,)
        avail += all_nodes - result[4][result[1]] - len(model3.authoritative_T)
    avg_avail_nodes_3.append(avail/500)
avg_avail_nodes_3
# %%
avg_avail_nodes_4 = []
all_nodes = nx.number_of_nodes(G4)
for i in np.linspace(0.01,0.05,9):
    avail = 0
    for j in range(500):
        model4.refresh_i_c_threshold()
        r_n = model4.generate_R_nodes(0,i)
        result = model4.before_detected_diffusion(r_n,)
        avail += all_nodes - result[4][result[1]] - len(model4.authoritative_T)
    avg_avail_nodes_4.append(avail/500)
avg_avail_nodes_4
# %%
avg_avail_rate_1 = np.array(avg_avail_nodes_1)/nx.number_of_nodes(G1)
avg_avail_rate_2 = np.array(avg_avail_nodes_2)/nx.number_of_nodes(G2)
avg_avail_rate_3 = np.array(avg_avail_nodes_3)/nx.number_of_nodes(G3)
avg_avail_rate_4 = np.array(avg_avail_nodes_4)/nx.number_of_nodes(G4)

fig,axs = plt.subplots(1,1,figsize=[8,5],facecolor='w')
x = np.linspace(0.01,0.05,9)
axs.plot(x,avg_avail_rate_1,label='dolphin',marker='o')
axs.plot(x,avg_avail_rate_2,label='food',marker='x')
axs.plot(x,avg_avail_rate_3,label='netscience',marker='*')
axs.plot(x,avg_avail_rate_4,label='USpower',marker='^')
axs.set_xlabel('Rumor nodes percent')
axs.set_ylabel('residual available nodes')
plt.legend()
plt.grid(True)
plt.title('(fixed authoritative T)authoritative T percent: 0.004')
plt.savefig('../res_pic/avg_residual_avail_nodes_1.tif',dpi=300)
plt.show()
# %%
