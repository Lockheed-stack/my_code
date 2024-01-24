# %%
import torch
import torch.nn as nn
from torch import optim
import networkx as nx
# %%


class A2C(nn.Module):
    """
    (Synchronous) Advantage Actor-Critic agent class

    Args:
        n_features: The number of features of the input state.
        n_actions: The number of actions the agent can take.
        device: The device to run the computations on.
        critic_lr: The learning rate for the critic network (should usually be larger than the actor_lr).
        actor_lr: The learning rate for the actor network.
        n_envs: The number of environments that run in parallel (on multiple CPUs) to collect experiences.
    """

    def __init__(
            self,
            G: nx.Graph,
            n_features: int,
            n_actions: int,
            hidden1:int,
            hidden2:int,
            device: torch.device,
            critic_lr: float,
            actor_lr: float,
            n_envs: int
    ):
        super().__init__()

        # Graph relate
        self.G = G.copy()
        self.node_num = G.number_of_nodes()

        # actor & critic network define
        self.device = device
        self.n_envs = n_envs
        # self.commonLayer = nn.Sequential(
        #     nn.Linear(n_features, hidden1),
        #     nn.ReLU(),
        #     nn.Linear(hidden1, hidden2),
        #     nn.ReLU(),
        # )
        self.critic = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)  # estimate V(s)
        ).to(self.device)
        self.actor = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(
                hidden2, n_actions
            ),  # estimate action logits (will be fed into a softmax later)
        ).to(self.device)

        self.initialize()
        
        # define optimizers
        self.critic_optim = optim.RMSprop(
            self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=actor_lr)

    def initialize(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight.data)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the networks.

        Args:
            x: A batched vector of states.

        Returns:
            state_values: A tensor with the state values, with shape [n_envs,].
            action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
        """
        state_values = self.critic(x)  # shape: [n_envs,]
        action_logits_vec = self.actor(x)  # shape:[n_envs,n_actions]
        return (state_values, action_logits_vec)

    def select_action(self, x: torch.Tensor, nodeState:list):
        """
        Returns a tuple of the chosen actions and the log-probs of those actions.

        Args:
            x: A batched vector of states.

        Returns:
            actions: A tensor with the actions, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
            state_values: A tensor with the state values, with shape [n_steps_per_update, n_envs].
        """

        state_values, action_logits = self.forward(x)
        valid_action_logits = torch.zeros(
            (self.n_envs, self.node_num), device=self.device)

        valid_action_logits_mask = []
        for i in range(self.n_envs):
            valid_action_logits_mask.append(
                torch.tensor(
                    tuple(
                        map(lambda n: n not in nodeState[i], self.G.nodes())
                    ),
                    device=self.device,
                    dtype=torch.bool)
            )
        valid_action_logits_mask = torch.stack(
            valid_action_logits_mask).to(self.device)
        valid_action_logits[valid_action_logits_mask] = action_logits[valid_action_logits_mask]

        action_pd = torch.distributions.Categorical(
            logits=valid_action_logits
        )  # implicitly uses softmax
        actions = action_pd.sample()  # the actions is the index of node
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()

        return (actions, action_log_probs, state_values, entropy)

    def get_losses(
            self,
            rewards: torch.Tensor,
            action_log_probs: torch.Tensor,
            value_preds: torch.Tensor,
            entropy: torch.Tensor,
            masks: torch.Tensor,
            gamma: float,
            lam: float,
            ent_coef: float,
            device: torch.device,):
        '''
        Args:
            rewards: A tensor with the rewards for each time step in the episode, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions taken at each time step in the episode, with shape [n_steps_per_update, n_envs].
            value_preds: A tensor with the state value predictions for each time step in the episode, with shape [n_steps_per_update, n_envs].
            masks: A tensor with the masks for each time step in the episode, with shape [n_steps_per_update, n_envs].
            gamma: The discount factor.
            lam: The GAE hyperparameter. (lam=1 corresponds to Monte-Carlo sampling with high variance and no bias,
                                          and lam=0 corresponds to normal TD-Learning that has a low variance but is biased
                                          because the estimates are generated by a Neural Net).
            device: The device to run the computations on (e.g. CPU or GPU).

        Returns:
            critic_loss: The critic loss for the minibatch.
            actor_loss: The actor loss for the minibatch.
        '''

        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device=device)

        # compute the advantages using GAE
        # 这个 TD-error 就是 DQN 中的那个损失函数
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rewards[t] + gamma * masks[t] *
                value_preds[t + 1] - value_preds[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration
        actor_loss = (
            -(advantages.detach() * action_log_probs).mean() -
            ent_coef * entropy.mean()
        )
        return (critic_loss, actor_loss)

    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        """
        Updates the parameters of the actor and critic networks.

        Args:
            critic_loss: The critic loss.
            actor_loss: The actor loss.
        """
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
#%%

class Actor(nn.Module):
    def __init__(
            self,
            G: nx.Graph,
            n_features: int,
            n_actions: int,
            hidden1:int,
            hidden2:int,
            device: torch.device,
            n_envs: int
    ):
        super().__init__()

        # Graph relate
        self.G = G.copy()
        self.node_num = G.number_of_nodes()

        # actor network define
        self.device = device
        self.n_envs = n_envs

        self.actor = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(
                hidden2, n_actions
            ),  # estimate action logits (will be fed into a softmax later)
        ).to(self.device)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the networks.

        Args:
            x: A batched vector of states.

        Returns:
            state_values: A tensor with the state values, with shape [n_envs,].
            action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
        """
        action_logits_vec = self.actor(x)  # shape:[n_envs,n_actions]
        return action_logits_vec

    def select_action(self, x: torch.Tensor, nodeStatus: list):
        """
        Returns a tuple of the chosen actions and the log-probs of those actions.

        Args:
            x: A batched vector of states.

        Returns:
            actions: A tensor with the actions, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
            state_values: A tensor with the state values, with shape [n_steps_per_update, n_envs].
        """

        with torch.no_grad():
            action_logits = self.forward(x)
        
        valid_action_logits = torch.zeros(
            (self.n_envs, self.node_num), device=self.device)

        valid_action_logits_mask = []
        for i in range(self.n_envs):
            valid_action_logits_mask.append(
                torch.tensor(
                    tuple(
                        map(lambda n: n not in nodeStatus[i], self.G.nodes())
                    ),
                    device=self.device,
                    dtype=torch.bool)
            )
        valid_action_logits_mask = torch.stack(
            valid_action_logits_mask).to(self.device)
        valid_action_logits[valid_action_logits_mask] = action_logits[valid_action_logits_mask]

        action_pd = torch.distributions.Categorical(
            logits=valid_action_logits
        )  # implicitly uses softmax
        actions = action_pd.sample()  # the actions is the index of node
        actions = int(actions)

        return actions