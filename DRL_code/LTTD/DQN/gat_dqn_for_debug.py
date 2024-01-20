#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch_geometric.utils as utils
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros

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
class Custom_GATConv(MessagePassing):
    def __init__(
            self, 
            in_channels:int,
            out_channels:int,
            edge_index:torch.Tensor,
            heads:int=1,
            concat:bool=True,
            negative_slope:float=0.2,
            dropout:float=0.0,
            bias:bool = True
        ):
        super().__init__(node_dim=1,aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_index,_ = utils.add_self_loops(edge_index,)


        self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
        self.lin_dst = self.lin_src

        # The learnable parameters to compute attention coefficients:
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        # Transform nodeState to embedding
        self.lin_nodeState = nn.Linear(1,out_channels)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()

        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x:torch.Tensor, nodeSataus:torch.Tensor,
                return_attention_weights:bool=False):
        

        x_nodeState = self.lin_nodeState(nodeSataus)
        x = x+x_nodeState

        # standard GAT forward operation
        H,C = self.heads,self.out_channels
        
        Batch_size = x.shape[0]
        x_src = x_dst = self.lin_src(x).view(Batch_size,-1,H,C)
        x = (x_src,x_dst)

        alpha_src = (x_src*self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)
        
        alpha = self.edge_updater(self.edge_index, alpha=alpha,)
        out = self.propagate(self.edge_index,x=x,alpha=alpha)

        if self.concat:
            out = out.view(Batch_size,-1,self.heads*self.out_channels)
        else:
            out = out.mean(dim=2)
        
        if self.bias is not None:
            out = out + self.bias

        out  = F.relu(out) # already pass the output into activation function
        if return_attention_weights:
            return out, (self.edge_index,alpha)
        else:
            return out

    def edge_update(self,alpha_j:torch.Tensor,alpha_i:torch.Tensor,
                    index: torch.Tensor, ptr: torch.Tensor,
                    size_i: int):
        
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        
        if index.numel() == 0:
            return alpha

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = utils.softmax(alpha, index,dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha
    
    def message(self,x_j:torch.Tensor,alpha):
        return  alpha.unsqueeze(-1)*x_j
#%%
class DeepQNet(nn.Module):
    def __init__(
            self, 
            node_num:int,
            strucEmb:torch.Tensor,
            embedding_dim: int, 
            edge_index: torch.Tensor,
            heads:int=1,
            concat:bool=False,
            negative_slope:float=1e-2,
            device:str='cuda',
        ) -> None:
        super(DeepQNet,self).__init__()

        self.node_num = node_num
        self.strucEmb = strucEmb.clone()
        self.embedding_dim = embedding_dim
        self.edge_index = edge_index
        self.device = device

        self.lin1 = nn.Linear(embedding_dim, embedding_dim)
        self.lin2 = nn.Linear(embedding_dim, embedding_dim)
        self.lin3 = nn.Linear(2*embedding_dim, 1)

        self.gat1 = Custom_GATConv(embedding_dim,embedding_dim,edge_index,heads,concat,negative_slope)
        self.gat2 = Custom_GATConv(embedding_dim,embedding_dim,edge_index,heads,concat,negative_slope)
        self.gat3 = Custom_GATConv(embedding_dim,embedding_dim,edge_index,heads,concat,negative_slope)

    def forward(self,state:torch.Tensor,action:torch.Tensor=None,is_for_learn:bool=False):
        '''
        note: the action should be the index of node(NOT NODE ID!)
        '''
        reshaped_state = state.reshape(state.shape[0],self.node_num,1)
        x = self.gat1(self.strucEmb,reshaped_state)
        x = self.gat2(x,reshaped_state)
        x = self.gat3(x,reshaped_state)

        state_emb = torch.sum(x,1)
        beta_state = self.lin1(state_emb)
        
        if is_for_learn:
            beta_action = self.lin2(x)
            expand_state = torch.repeat_interleave(
                beta_state,
                torch.full((beta_action.shape[0],),self.node_num,device=self.device),
                dim=0,
            ).reshape(beta_action.shape[0],self.node_num,-1)
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
            embeding_dim: int,
            edge_index: torch.Tensor,
            heads:int=1,
            negative_slope:float=1e-2,
    ):


        # Graph relate settings
        self.G = G.copy()
        self.node_id_index_map = dict() # map node id to index
        self.node_index_id_map = dict()
        for i, node in enumerate(G.nodes()):
            self.node_id_index_map[node] = i
            self.node_index_id_map[i] = node
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

        self.memory = ReplayMemory(memory_capacity)

        self.embeding_dim = embeding_dim
        self.edge_index = edge_index
        self.strucEmb = pretrain_emb
        self.added_strucEmb = torch.sum(pretrain_emb,dim=0)


        # DeepQNet relate settings
        self.policy_net = DeepQNet(
            self.node_num,pretrain_emb,embeding_dim, edge_index,
            heads=heads,negative_slope=negative_slope,device=device).to(device)
        self.target_net = DeepQNet(self.node_num,pretrain_emb,embeding_dim, edge_index,
            heads=heads,negative_slope=negative_slope,device=device).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optim = optim.Adam(
            self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.loss_fn = nn.SmoothL1Loss()

    def choose_action(self, candidateNodes: set,state:torch.Tensor):
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
        
        state_action_values = self.policy_net(state_batch, action_batch)
        next_state_action_values = torch.zeros(
            (self.batch_size, 1), device=self.device)
        
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
        with torch.no_grad():
            result = self.target_net(non_final_next_states,None,is_for_learn=True)

        result[invalid_action_mask] = float('-inf')
        next_state_action_values[not_none_state_mask] = result.max(1)[0].view(result.shape[0])


        # compute the expected Q values
        expected_state_action_values = (
            next_state_action_values*self.gamma) + reward_batch

        # computer loss
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        # optimize the model
        self.optim.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(),100)
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
                action = dqn_model.choose_action(candidate_nodes, state)

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
                    episode_score.append(
                        len(info['T_active'])/len(info['R_active']))
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
HEADS = 1
NEGATIVE_SLOPE = 1e-2
MAX_MEMORY_CAPACITY = 50000

# %%
env1 = gym.make("LTTD-v0", G=G1, init_rumor_rate=0.1,
                au_T_rate=0.08, k_budget=0.15, alpha=0.8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
torch_geo_G1_data = utils.from_networkx(G1)
# %%
with open('pre_trained_GAE_emb/dol_pretrained_emb', 'rb') as f:
    pre_trained_emb = pickle.load(f,)

pre_trained_emb = pre_trained_emb.to(device)
# %%
dol_dqn = DQN(G1, device, MAX_MEMORY_CAPACITY, LEARNING_RATE_DOL, BATCH_SIZE_DOL,
              EPS_DECAY_DOL, EPS_START_DOL, EPS_END_DOL, GAMMA_DOL,
              pre_trained_emb, 32, torch_geo_G1_data.edge_index.to(device),
              HEADS,NEGATIVE_SLOPE)
# %%
episode_score_dol = []
episode_loss_dol = []
# %%
training(dol_dqn, env1, NODE_NUM_DOL, episode_score_dol,
         episode_loss_dol, 0.15, device,20000,0)
#%%
from torch.utils.tensorboard import SummaryWriter
#%%
writer = SummaryWriter(comment='_dol_k=015_head=1')

for i in range(len(episode_score_dol)):
    writer.add_scalar('GAT_DQN/Score',episode_score_dol[i],i+1)
for i in range(len(episode_loss_dol)):
    writer.add_scalar('GAT_DQN/Loss',episode_loss_dol[i],i+1)
writer.close()
#%%
torch.save(dol_dqn.target_net.state_dict(),'dol_gat_dqn_test')