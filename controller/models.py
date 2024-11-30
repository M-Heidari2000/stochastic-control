import torch
import torch.nn as nn
from torch.distributions import Normal


class ObservationModel(nn.Module):
    """
        outputs mean for p(o_t | s_t) ~ N(f(s_t), I)
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
            nn.Linear(hidden_dim, observation_dim),
        )

    def forward(self, state):
        return self.mlp_layers(state)