#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch_geometric.utils as utils
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot

import pickle
import gym_LTTD_v0
import gymnasium as gym
from tqdm import tqdm
import networkx as nx
import numpy as np
from itertools import count
from collections import namedtuple, deque
import networkx as nx
import random
import math
#%%
class Custom_GATConv_v2(nn.Module):
    def __init__(
            self, 
            in_channels:int,
            out_channels:int,
            heads:int=1,
            negative_slope:float=0.2,
            dropout:float=0.0,
        ) -> None:
        super(Custom_GATConv_v2,self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.W = Linear(in_channels, heads * out_channels,
                            bias=False, weight_initializer='glorot')
        self.attention = nn.Parameter(torch.Tensor(heads, 2*out_channels,1))

        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        self.W.reset_parameters()
        glorot(self.attention)

    def _get_attention_scores(self,Wh:torch.Tensor):
        
        # neighbors message
        source_scores = torch.matmul(Wh,self.attention[:,:self.out_channels,:])
        # target nodes
        target_scores = torch.matmul(Wh,self.attention[:,self.out_channels:,:])

        e = source_scores + target_scores.mT

        return self.leakyrelu(e)

    def forward(self,x:torch.Tensor,adj_matrix_masked:torch.Tensor):
        
        Wh = self.W(x) # shape:(batch * nodes * (heads * out_channels))
        # from shape: 'batch * nodes * heads * out_channels' 
        # to 'batch * heads * nodes * out_channels'
        Wh = Wh.reshape(Wh.shape[0],Wh.shape[1],self.heads,self.out_channels).permute(0,2,1,3)

        e = self._get_attention_scores(Wh)
        e = torch.masked_fill(e,adj_matrix_masked,float('-inf'))

        attention = F.softmax(e,dim=-1)

        out = torch.matmul(attention,Wh)

        out = out.mean(dim=1)

        return out
#%%
class DeepQNet(nn.Module):
    def __init__(
            self, 
            node_num:int,
            strucEmb:torch.Tensor,
            embedding_dim: int, 
            adj_matrix: torch.Tensor,
            heads:int=1,
            negative_slope:float=0.2,
            device:str='cuda',
        ) -> None:
        super(DeepQNet,self).__init__()

        self.node_num = node_num
        self.strucEmb = strucEmb.clone()
        self.embedding_dim = embedding_dim
        self.adj_matrix_masked = ~adj_matrix.bool()
        self.device = device

        self.weight_struc = nn.Linear(embedding_dim,embedding_dim,False)
        self.weight_state = nn.Linear(embedding_dim,embedding_dim,False)
        self.lin1 = nn.Linear(embedding_dim, embedding_dim)
        self.lin2 = nn.Linear(embedding_dim, embedding_dim)
        self.lin3 = nn.Linear(2*embedding_dim, 1)

        self.gat1 = Custom_GATConv_v2(1,embedding_dim,heads,negative_slope)


    def forward(self,state:torch.Tensor,action:torch.Tensor=None,is_for_learn:bool=False):
        '''
        note: the action should be the index of node(NOT NODE ID!)
        '''
        # aggregate node state information
        reshaped_state = state.reshape(state.shape[0],self.node_num,1)
        x = self.gat1(reshaped_state,self.adj_matrix_masked)
        x = F.relu(self.weight_struc(self.strucEmb)+self.weight_state(x))

        state_emb = torch.sum(x,1)
        beta_state = self.lin1(state_emb)
        
        if is_for_learn:
            beta_action = self.lin2(x)
            expand_state =  beta_state.repeat(1,self.node_num).reshape(beta_action.shape[0],self.node_num,self.embedding_dim)
            out = nn.functional.relu(torch.cat([expand_state,beta_action], dim=2))
        else:
            action_emb = x[torch.arange(x.size(0)),action] # select coresponding action embedding
            beta_action = self.lin2(action_emb)
            # if is_same_state:
            #     beta_state = beta_state.repeat(beta_action.size(0),1)
            out = nn.functional.relu(torch.cat([beta_state,beta_action], dim=1))
            
        return self.lin3(out)
#%%
Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state'))


class ReplayMemory():
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#%%
class DQN:
    def __init__(
            self,
            G: nx.Graph,
            device: str,
            memory_capacity: int,
            LR: float,
            batch_size: int,
            eps_decay: int,
            eps_start: float,
            eps_end: float,
            gamma: float,
            pretrain_emb: torch.Tensor,
            embedding_dim: int,
            adj_matrix: torch.Tensor,
            heads:int=1,
            negative_slope:float=1e-2,
            using_recommend:float = 0.5,
            mixed_precision:bool=False,
    ):

        # enable mixed-precision training
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        # Graph relate settings
        self.G = G.copy()
        self.node_id_index_map = dict() # map node id to index
        for i, node in enumerate(G.nodes()):
            self.node_id_index_map[node] = i
        self.node_num = G.number_of_nodes()

        # hyperparameters settings
        self.device = device
        self.memory_capacity = memory_capacity
        # self.gather_exp_size = gather_exp_size
        self.batch_size = batch_size
        self.eps_decay = eps_decay
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.gamma = gamma
        self.learn_step_counter = 0
        self.using_recommend = using_recommend
        self.mixed_precision = mixed_precision
        self.memory = ReplayMemory(memory_capacity)

        self.embedding_dim = embedding_dim
        self.strucEmb = pretrain_emb
        self.added_strucEmb = torch.sum(pretrain_emb,dim=0)


        # DeepQNet relate settings
        self.policy_net = DeepQNet(self.node_num,pretrain_emb,embedding_dim, adj_matrix,
            heads=heads,negative_slope=negative_slope,device=device).to(device)
        self.target_net = DeepQNet(self.node_num,pretrain_emb,embedding_dim, adj_matrix,
            heads=heads,negative_slope=negative_slope,device=device).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optim = optim.AdamW(
            self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.loss_fn = nn.SmoothL1Loss()


    def choose_action(self, candidateNodes: set,recommend:set(),state:torch.Tensor):
        '''
        This function will return the index of coresponding node
        '''
        eps_threshold = self.eps_end + (self.eps_start-self.eps_end) * math.exp(-1.*self.learn_step_counter/self.eps_decay)

        if random.random() > eps_threshold:
            with torch.no_grad():
                state_mask = (state.bool()).reshape(self.node_num,1)
                
                result = self.policy_net(state,None,is_for_learn=True).squeeze(0)
                result[state_mask] = float('-inf')
                
                return int(result.max(0)[1])
        else:
            if len(recommend)>0 and (self.using_recommend > random.random()):
                action = list(recommend)
                return self.node_id_index_map[random.choice(action)]
            else:
                action = list(candidateNodes)
                return self.node_id_index_map[random.choice(action)]

    def learn(self):
        if len(self.memory) < self.batch_size:
            
            return None

        b_memory = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*b_memory))

        state_batch = torch.cat(batch.state)
        reward_batch = torch.cat(batch.reward)
        action_batch = torch.cat(batch.action)
        
        
        invalid_action_mask = [] # for masking invalid actions in coresponding state
        not_none_state_mask = [] # for updating next_state_action_values
        non_final_next_states = [] # for storing not-none next state
        for state in batch.next_state:
            if state is not None:
                non_final_next_states.append(state[0])
                invalid_action_mask.append(state[1])
                not_none_state_mask.append(True)
            else:
                not_none_state_mask.append(False)
        
        invalid_action_mask = torch.cat(invalid_action_mask).reshape(len(invalid_action_mask),self.node_num,1)
        not_none_state_mask = torch.tensor(not_none_state_mask,dtype=bool).reshape(self.batch_size,1)
        non_final_next_states = torch.cat(non_final_next_states,)
            
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    result = self.target_net(non_final_next_states,None,is_for_learn=True)
            
            state_action_values = self.policy_net(state_batch, action_batch)
            next_state_action_values = torch.zeros((self.batch_size, 1), device=self.device,dtype=torch.float16)

            result[invalid_action_mask] = torch.tensor(float('-inf'),dtype=torch.float16)
            next_state_action_values[not_none_state_mask] = result.max(1)[0].view(result.shape[0])

            # compute the expected Q values
            expected_state_action_values = (next_state_action_values*self.gamma) + reward_batch

            with torch.cuda.amp.autocast(): 
                loss = self.loss_fn(state_action_values, expected_state_action_values)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            self.optim.zero_grad()
        else:
            with torch.no_grad():
                result = self.target_net(non_final_next_states,None,is_for_learn=True)
            
            state_action_values = self.policy_net(state_batch, action_batch)
            next_state_action_values = torch.zeros((self.batch_size, 1), device=self.device)

            result[invalid_action_mask] = float('-inf')
            next_state_action_values[not_none_state_mask] = result.max(1)[0].view(result.shape[0])

            # compute the expected Q values
            expected_state_action_values = (next_state_action_values*self.gamma) + reward_batch

            # compute loss and backward
            loss = self.loss_fn(state_action_values, expected_state_action_values)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return loss
#%%
def print_training_result(current_episode: int, episode_score: list, episode_loss: list):
    avg_score = np.array(episode_score).mean()
    avg_loss = torch.tensor(episode_loss).mean()
    # print(f'\nepisode:{current_episode}, last 100 average score:{avg_score:.3f}, last 100 average_loss:{avg_loss:.4f}')
    return avg_score,avg_loss
#%%
def gather_replay_exp(
        dqn_model: DQN, 
        env: gym.Env,
        exp_num: int, 
        nodes_set:set,
        node_num: int,
        select_node_num:int,
        device:str,
    ):
    while (collected_exp_num:=len(dqn_model.memory)) < exp_num:
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for _ in count():
            nodeState = info['T_active'] | info['R_active']
            if node_num -len(nodeState) <= select_node_num :
                break
            
            candidate_nodes = nodes_set-nodeState
            action = dqn_model.choose_action(candidate_nodes, state)
            # candidate_nodes.discard(action)

            obs, reward, terminated, _, info = env.step(action)# the env need the index of node
            reward = torch.tensor([[reward]], device=device, dtype=torch.float32)
            # need node index to select embeddings
            action = torch.tensor([action], device=device)
            done = terminated

            if terminated:
                next_state = None
                dqn_model.memory.push(state, action, reward, next_state)
            else:
                next_state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                dqn_model.memory.push(state, action, reward, [next_state,~next_state.bool()])

            # move to the next state
            state = next_state

            if done:
                break
        print(f'collecting exp:{collected_exp_num} ...\r', end='')
#%%
def training(
        dqn_model: DQN, 
        env: gym.Env, 
        node_num: int, 
        episode_score: list, 
        episode_loss: list, 
        select_k_T:float,
        device:str,
        num_episode: int = 5000,
        exp_gather_num:int = 5000,
        ):

    select_K_Tnodes = round(node_num*select_k_T)
    nodes_set = set(dqn_model.G.nodes())
    
    if exp_gather_num >0:
        gather_replay_exp(dqn_model,env,exp_gather_num,nodes_set,node_num,select_K_Tnodes,device)

    
    with tqdm(total=num_episode,desc='episode:Nan, score:Nan, loss:Nan,') as pbar:
        for i_episode in range(1, num_episode + 1):
            # initialize the environment and get is's state
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            dqn_model.learn_step_counter += 1
            for _ in count():
                nodeState = info['T_active'] | info['R_active']
                if node_num -len(nodeState) <= select_K_Tnodes :
                    break
                
                candidate_nodes = nodes_set-nodeState
                action = dqn_model.choose_action(candidate_nodes,info['recommend'],state)

                obs, reward, terminated, _, info = env.step(action)# the env need the index of node
                reward = torch.tensor([[reward]], device=device, dtype=torch.float32)
                # need node index to select embeddings
                action = torch.tensor([action], device=device)
                done = terminated

                if terminated:
                    next_state = None
                    dqn_model.memory.push(state, action, reward, next_state)
                else:
                    next_state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    dqn_model.memory.push(state, action, reward, [next_state,next_state.bool()])

                # move to the next state
                state = next_state

                loss = dqn_model.learn()
                if done:
                    episode_score.append(len(info['T_active'])/len(info['R_active']))
                    if loss is not None:
                        episode_loss.append(loss)
                    break

            # target_net_state_dict = karate_dqn.target_net.state_dict()
            # policy_net_state_dict = karate_dqn.policy_net.state_dict()
            # for key in policy_net_state_dict:
            #     target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            # karate_dqn.target_net.load_state_dict(target_net_state_dict)

            if i_episode % 100 == 0:
                res = print_training_result(i_episode, episode_score, episode_loss)
                pbar.set_description(f'episode:{i_episode}, score:{res[0]:.3f}, loss:{res[1]:.3f}')
                pbar.update(100)
            if i_episode % 100 == 0:
                dqn_model.target_net.load_state_dict(dqn_model.policy_net.state_dict())

    print("Complete")
#%%
# ------------------------------ Dolphins -------------------------------------------
G1 = nx.read_adjlist('../dataset/dolphins.mtx', nodetype=int)
# %%
BATCH_SIZE_DOL = 64
NODE_NUM_DOL = G1.number_of_nodes()
EPS_DECAY_DOL = 10000
EPS_START_DOL = 0.99
EPS_END_DOL = 0.05
GAMMA_DOL = 0.9
LEARNING_RATE_DOL = 1e-3
HEADS = 4
NEGATIVE_SLOPE = 0.2
MAX_MEMORY_CAPACITY = 50000
USING_RECOMMEND = 0.5
# %%
env1 = gym.make("LTTD-v0", G=G1, init_rumor_rate=0.1,
                au_T_rate=0.08, k_budget=0.15,target_to_reach=5.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
torch_geo_G1_data = utils.from_networkx(G1)
edge_index,_ = utils.add_self_loops(torch_geo_G1_data.edge_index)
adj_matrix = torch.zeros((NODE_NUM_DOL,NODE_NUM_DOL))
adj_matrix[edge_index[0],edge_index[1]]=1
# %%
with open('pre_trained_GAE_emb/dol_pretrained_emb', 'rb') as f:
    pre_trained_emb = pickle.load(f,)

pre_trained_emb = pre_trained_emb.to(device)
# %%
dol_dqn = DQN(G1, device, MAX_MEMORY_CAPACITY, LEARNING_RATE_DOL, BATCH_SIZE_DOL,
              EPS_DECAY_DOL, EPS_START_DOL, EPS_END_DOL, GAMMA_DOL,
              pre_trained_emb, 32, adj_matrix.to(device),
              HEADS,NEGATIVE_SLOPE,USING_RECOMMEND)
# %%
episode_score_dol = []
episode_loss_dol = []
# %%
training(dol_dqn, env1, NODE_NUM_DOL, episode_score_dol,
         episode_loss_dol, 0.15, device,20000,0)
# #%%
from torch.utils.tensorboard import SummaryWriter
#%%
writer = SummaryWriter(comment='_dol_k=015_head=4_gatv2')

for i in range(len(episode_score_dol)):
    writer.add_scalar('GAT_DQN/Score',episode_score_dol[i],i+1)
for i in range(len(episode_loss_dol)):
    writer.add_scalar('GAT_DQN/Loss',episode_loss_dol[i],i+1)
writer.close()
#%%
torch.save(dol_dqn.target_net.state_dict(),'dol_gat_dqn_test')
#%%
