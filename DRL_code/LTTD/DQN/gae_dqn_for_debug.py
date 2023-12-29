# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.utils as utils
from torch_geometric.nn import MessagePassing
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
# %%
class Custom_GCN(MessagePassing):
    def __init__(self, emb_dim: int, edge_index: torch.Tensor) -> None:
        super().__init__(aggr='add')

        self.edge_index, _ = utils.add_self_loops(edge_index)
        self.lin_nbr = nn.Linear(emb_dim, emb_dim)
        self.lin_nodeState = nn.Linear(1, emb_dim)

    def forward(self, nodeEmb: torch.Tensor, nodeStatus: torch.Tensor):

        x0 = self.propagate(self.edge_index, x=nodeEmb)
        x0 = self.lin_nbr(x0)
        x1 = self.lin_nodeState(nodeStatus)
        return nn.functional.relu(x0+x1)
    # def propagate_only(self,nodeEmb:torch.Tensor):
    #     return self.propagate(self.edge_index,x=nodeEmb)
# %%
class nerual_network(nn.Module):
    def __init__(self, node_num:int,strucEmb:torch.Tensor,embeding_dim: int, edge_index: torch.Tensor) -> None:
        super(nerual_network, self).__init__()
        self.node_num = node_num
        self.strucEmb = strucEmb.clone()
        self.edge_index = edge_index

        self.lin1 = nn.Linear(embeding_dim, embeding_dim)
        self.lin2 = nn.Linear(embeding_dim, embeding_dim)
        self.lin3 = nn.Linear(2*embeding_dim, 1)
        
        # self.gcn1 = GCNConv(embeding_dim,embeding_dim)
        self.gcn1 = Custom_GCN(embeding_dim, edge_index)
        self.gcn2 = Custom_GCN(embeding_dim, edge_index)
        self.gcn3 = Custom_GCN(embeding_dim, edge_index)

    def forward(self,state: torch.Tensor, action: torch.Tensor,is_same_state:bool=False,is_for_learn:bool=False):
        '''
        note: the action should be the index of node(NOT NODE ID!)
        '''
        reshaped_state = state.reshape(state.shape[0],self.node_num,1)
        x = self.gcn1(self.strucEmb,reshaped_state)
        x = self.gcn2(x,reshaped_state)
        x = self.gcn3(x,reshaped_state)


        state_emb = torch.sum(x,1)
        beta_state = self.lin1(state_emb)
        
        if is_for_learn:
            beta_action = self.lin2(x)
            expand_state = []
            for s in beta_state:
                expand_state.append(s.repeat(beta_action.shape[1],1))
            expand_state = torch.stack(expand_state)
            out = nn.functional.relu(torch.cat([expand_state,beta_action], dim=2))
        else:
            action_emb = x[torch.arange(x.size(0)),action] # select coresponding action embedding
            beta_action = self.lin2(action_emb)
            if is_same_state:
                beta_state = beta_state.repeat(beta_action.size(0),1)
            out = nn.functional.relu(torch.cat([beta_state,beta_action], dim=1))
            
        return self.lin3(out)

# %%
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
    ):

        self.G = G.copy()
        self.node_id_index_map = dict() # map node id to index
        self.node_index_id_map = dict()
        for i, node in enumerate(G.nodes()):
            self.node_id_index_map[node] = i
            self.node_index_id_map[i] = node
        self.node_num = G.number_of_nodes()

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
        self.strucEmb = pretrain_emb
        self.added_strucEmb = torch.sum(pretrain_emb,dim=0)

        self.policy_net, self.target_net = nerual_network(self.node_num,pretrain_emb,embeding_dim, edge_index).to(
            device), nerual_network(self.node_num,pretrain_emb,embeding_dim, edge_index).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optim = optim.Adam(
            self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.loss_fn = nn.SmoothL1Loss()
    # @profile
    def choose_action(self, candidateNodes: set,state:torch.Tensor):
        '''
        This function will return the index of coresponding node
        '''
        eps_threshold = self.eps_end + (self.eps_start-self.eps_end) * math.exp(-1.*self.learn_step_counter/self.eps_decay)

        if random.random() > eps_threshold:
            with torch.no_grad():
                # node_index_0 = tuple(map(lambda n:self.node_id_index_map[n],candidateNodes))
                # sorted_index = sorted(node_index_0)
                node_index = []
                for i,node in enumerate(state[0]):
                    if node==0:
                        node_index.append(i)
                
                max_Q_action_index = self.policy_net(
                    state, 
                    # torch.tensor(sorted_index),
                    torch.tensor(node_index),
                    True
                ).max(0)[1]
                # return self.node_index_id_map[sorted_index[int(max_Q_action_index)]]
                return node_index[int(max_Q_action_index)]
        else:
            action = list(candidateNodes)
            return self.node_id_index_map[random.choice(action)]
    # @profile
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
        # foo = torch.zeros(
        #     (self.batch_size, 1), device=self.device)
        # for i, state_with_action in enumerate(batch.next_state):
        #     if state_with_action is not None:
        #         with torch.no_grad():
        #             actions = torch.tensor(tuple(map(lambda n:self.node_id_index_map[n],state_with_action[2])))
        #             foo[i] = self.target_net(state_with_action[0], actions,is_same_state=True).max()
        
        valid_action_mask = [] # for selecting valid actions in coresponding state
        not_none_state_mask = [] # for updating next_state_action_values
        non_final_next_states = [] # for storing not-none next state
        for state in batch.next_state:
            if state is not None:
                non_final_next_states.append(state[0])
                valid_action_mask.append(state[1])
                not_none_state_mask.append(True)
            else:
                not_none_state_mask.append(False)
        
        valid_action_mask = torch.cat(valid_action_mask).reshape(len(valid_action_mask),self.node_num,1)
        not_none_state_mask = torch.tensor(not_none_state_mask,dtype=bool).reshape(self.batch_size,1)
        non_final_next_states = torch.cat(non_final_next_states,)
        with torch.no_grad():
            result = self.target_net(non_final_next_states,None,is_for_learn=True)
        
        target_net_result = []
        for i in range(result.shape[0]):
            target_net_result.append(result[i,valid_action_mask[i]].max())

        target_net_result = torch.hstack(target_net_result)
        next_state_action_values[not_none_state_mask] = target_net_result

        
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

    # gather_replay_exp(dqn_model,env,exp_gather_num,nodes_set,node_num,select_K_Tnodes,device)

    
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
                candidate_nodes.discard(action)

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
BATCH_SIZE_DOL = 5
NODE_NUM_DOL = G1.number_of_nodes()
EPS_DECAY_DOL = 10000
EPS_START_DOL = 0.99
EPS_END_DOL = 0.05
GAMMA_DOL = 0.9
LEARNING_RATE_DOL = 1e-3
MAX_MEMORY_CAPACITY = 10000

# %%
env1 = gym.make("LTTD-v0", G=G1, init_rumor_rate=0.1,
                au_T_rate=0.08, k_budget=0.05, alpha=0.8)
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
              pre_trained_emb, 32, torch_geo_G1_data.edge_index.to(device))
# %%
episode_score_dol = []
episode_loss_dol = []
# %%
# torch.autograd.set_detect_anomaly(True)
training(dol_dqn, env1, NODE_NUM_DOL, episode_score_dol,
         episode_loss_dol, 0.05, device,20,5000)
# %%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment='_test')

for i in range(len(episode_score_dol)):
    writer.add_scalar('GAE_DQN/Score',episode_score_dol[i],i+1)
for i in range(len(episode_loss_dol)):
    writer.add_scalar('GAE_DQN/Loss',episode_loss_dol[i],i+1)
#%%
    