import torch
import torch.nn as nn
from typing import Optional
from torch.distributions import Normal


class ObservationModel(nn.Module):
    """
        p(o_t | s_t)
        constant variance
    """
    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        hidden_dim: Optional[int]=None,
    ):
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim is not None else state_dim * 2
        
        self.mlp_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, observation_dim),
        )

    def forward(self, state):
        return self.mlp_layers(state)
    

class RewardModel(nn.Module):
    """
        p(r_t | s_t)
        constant variance
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dim: Optional[int]=None,
    ):
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim is not None else 2*state_dim

        self.mlp_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.mlp_layers(state)
    

class TransitionModel(nn.Module):
    """
        p(s_t | s_{t-1}, a_{t-1})
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: Optional[int]=None,
        min_std: float=1e-2,
    ):
        super().__init__()

        hidden_dim = (
            hidden_dim if hidden_dim is not None else 2*(state_dim + action_dim)
        )

        self.mlp_layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, state_dim)
        self.std_head = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus(),
        )

        self._min_std = min_std

    def forward(self, prev_state, prev_action):
        hidden = self.mlp_layers(
            torch.cat([prev_state, prev_action], dim=1)
        )
        mean = self.mean_head(hidden)
        std = self.std_head(hidden) + self._min_std
        prior_dist = Normal(mean, std)
        return prior_dist
    

class PosteriorModel(nn.Module):
    """
        q(s_t | o1:t, a1:t-1)
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        state_dim: int,
        rnn_hidden_dim: int,
        rnn_input_dim: int,
        min_std: float=1e-2,
    ):
        super().__init__()

        # RNN hidden at time t summarizes o1:t-1, a1:t-2
        self.rnn = nn.GRUCell(
            input_size=rnn_input_dim,
            hidden_size=rnn_hidden_dim,
        )

        self.fc_obs_action = nn.Sequential(
            nn.Linear(observation_dim + action_dim, rnn_input_dim),
            nn.ReLU(),
        )

        self.posterior_mean_head = nn.Linear(rnn_hidden_dim, state_dim)
        self.posterior_std_head = nn.Sequential(
            nn.Linear(rnn_hidden_dim, state_dim),
            nn.Softplus(),
        )

        self.rnn_hidden_dim = rnn_hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rnn_input_dim = rnn_input_dim
        self._min_std = min_std

    def forward(
        self,
        prev_rnn_hidden,
        prev_action,
        observation,
    ):
        rnn_input = self.fc_obs_action(
            torch.cat([observation, prev_action], dim=1)
        )
        rnn_hidden = self.rnn(rnn_input, prev_rnn_hidden)

        posterior_mean = self.posterior_mean_head(rnn_hidden)
        posterior_std = self.posterior_std_head(rnn_hidden) + self._min_std

        posterior_dist = Normal(posterior_mean, posterior_std)

        return rnn_hidden, posterior_dist