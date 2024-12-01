import torch
import einops
from typing import Callable
from torch.distributions import Normal


class CEMAgent:
    """
        action planning by Cross Entropy Method (CEM)
    """

    def __init__(
        self,
        transition_model,
        posterior_model,
        planning_horizon: int,
        num_iterations: int,
        num_candidates: int,
        num_elites: int,
        reward_function: Callable,
    ):
        self.transition_model = transition_model
        self.posterior_model = posterior_model
        self.num_iterations = num_iterations
        self.num_candidates = num_candidates
        self.num_elites = num_elites
        self.planning_horizon = planning_horizon
        self.reward_function = reward_function

        self.device = next(posterior_model.parameters()).device

        # initialize rnn hidden to zero vector
        self.rnn_hidden = torch.zeros(1, self.posterior_model.rnn_hidden_dim, device=self.device)

    def __call__(self, obs, prev_action=None):

        # convert o_t and a_{t-1} to a torch tensor and add a batch dimension
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        if prev_action is not None:
            prev_action = torch.as_tensor(prev_action, device=self.device).unsqueeze(0)
        else:
            prev_action = torch.zeros(self.posterior_model.action_dim, device=self.device).unsqueeze(0)

        # no learning takes place here
        with torch.no_grad():
            # infer s_t using q(s_t | o1:t, a1:t-1)
            self.rnn_hidden, state_posterior = self.posterior_model(
                prev_rnn_hidden=self.rnn_hidden,
                prev_action=prev_action,
                observation=obs,
            )

            # initialize action distribution ~ N(0, I)
            action_dist = Normal(
                torch.zeros((self.planning_horizon, self.posterior_model.action_dim), device=self.device),
                torch.ones((self.planning_horizon, self.posterior_model.action_dim), device=self.device),
            )

            # iteratively improve action distribution with CEM
            for _ in range(self.num_iterations):
                # sample action candidates
                # reshape to (planning_horizon, num_candidates, action_dim) for parallel exploration
                action_candidates = action_dist.sample([self.num_candidates])
                action_candidates = einops.rearrange(action_candidates, "n h a -> h n a")

                state = state_posterior.sample([self.num_candidates]).squeeze(-2)
                total_predicted_reward = torch.zeros(self.num_candidates, device=self.device)

                # start generating trajectories starting from s_t using transition model
                for t in range(self.planning_horizon):
                    total_predicted_reward += self.reward_function(
                        state=state,
                        action=action_candidates[t],
                    ).squeeze()
                    # get next state from our prior (transition model)
                    next_state_prior = self.transition_model(
                        prev_state=state,
                        prev_action=action_candidates[t],
                    )
                    state = next_state_prior.sample()

                # find elites
                elite_indexes = total_predicted_reward.argsort(descending=True)[:self.num_elites]
                elites = action_candidates[:, elite_indexes, :]

                # fit a new distribution to the elites
                mean = elites.mean(dim=1)
                std = elites.std(dim=1, unbiased=False)
                action_dist.loc = mean
                action_dist.scale = std
            
            #return only mean of the first action (MPC)
            action = mean[0]

        return action.cpu().numpy()
    
    def reset(self):
        self.rnn_hidden = torch.zeros(1, self.posterior_model.rnn_hidden_dim, device=self.device)