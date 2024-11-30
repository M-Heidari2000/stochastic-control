import torch
import torch.nn as nn
from torch.distributions import Normal


class ObservationModel(nn.Module):
    """
        p(o_t | s_t)
        constant variance
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        observation_dim: int,
    ):
        super().__init__()
        
        self.mlp_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, observation_dim),
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
        hidden_dim: int,
    ):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, state_dim)
        self.log_std_head = nn.Linear(hidden_dim, state_dim)

    def forward(self, prev_state, prev_action):
        hidden = self.mlp_layers(
            torch.cat([prev_state, prev_action], dim=1)
        )
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden)
        return Normal(mean, log_std.exp())