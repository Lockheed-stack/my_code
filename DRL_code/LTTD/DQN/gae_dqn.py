import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.utils as utils
from torch_geometric.nn import MessagePassing
import networkx as nx
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

    def forward(self,state: torch.Tensor, action: torch.Tensor,is_same_state:bool=False):
        '''
        note: the action should be the index of node(NOT NODE ID!)
        '''
        reshaped_state = state.reshape(state.shape[0],self.node_num,1)
        x = self.gcn1(self.strucEmb,reshaped_state)
        x = self.gcn2(x,reshaped_state)
        x = self.gcn3(x,reshaped_state)


        state_emb = torch.sum(x,1)
        action_emb = x[torch.arange(x.size(0)),action] # select coresponding action embedding

        beta_state = self.lin1(state_emb)
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

    def choose_action(self, candidateNodes: set,state:torch.Tensor):

        eps_threshold = self.eps_end + (self.eps_start-self.eps_end) * math.exp(-1.*self.learn_step_counter/self.eps_decay)

        if random.random() > eps_threshold:
            with torch.no_grad():
                node_index = tuple(map(lambda n:self.node_id_index_map[n],candidateNodes))
                sorted_index = sorted(node_index)
                
                max_Q_action_index = self.policy_net(
                    state, 
                    torch.tensor(sorted_index),
                    True
                ).max(0)[1]
                return self.node_index_id_map[sorted_index[int(max_Q_action_index)]]
        else:
            action = list(candidateNodes)
            return random.choice(action)

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
        for i, state_with_action in enumerate(batch.next_state):
            if state_with_action is not None:
                with torch.no_grad():
                    actions = torch.tensor(tuple(map(lambda n:self.node_id_index_map[n],state_with_action[1])))
                    next_state_action_values[i] = self.target_net(state_with_action[0], actions,True).max()
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
class DQN_FOR_TEST(nn.Module):
    def __init__(
            self, 
            G: nx.Graph,
            device: str,
            pretrain_emb: torch.Tensor,
            embeding_dim: int,
            edge_index: torch.Tensor,
        ) -> None:
        super().__init__()

        self.G = G.copy()
        self.node_id_index_map = dict() # map node id to index
        self.node_index_id_map = dict()
        for i, node in enumerate(G.nodes()):
            self.node_id_index_map[node] = i
            self.node_index_id_map[i] = node
        self.node_num = G.number_of_nodes()

        self.device = device
        self.policy_net = nerual_network(self.node_num,pretrain_emb,embeding_dim, edge_index).to(device)
    def choose_action(self, candidateNodes: set,state:torch.Tensor):

        with torch.no_grad():
            node_index = tuple(map(lambda n:self.node_id_index_map[n],candidateNodes))
            sorted_index = sorted(node_index)
            
            max_Q_action_index = self.policy_net(
                state, 
                torch.tensor(sorted_index),
                True
            ).max(0)[1]
            return self.node_index_id_map[sorted_index[int(max_Q_action_index)]]