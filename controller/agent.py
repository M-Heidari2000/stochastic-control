import torch
import einops
from torch.distributions import Normal, Uniform


class CEMAgent:
    """
        action planning by Cross Entropy Method (CEM)
    """
    def __init__(
        self,
        transition_model,
        posterior_model,
        reward_model,
        observation_model,
        planning_horizon: int,
        num_iterations: int,
        num_candidates: int,
        num_elites: int,
    ):
        self.transition_model = transition_model
        self.posterior_model = posterior_model
        self.reward_model = reward_model
        self.observation_model = observation_model
        self.num_iterations = num_iterations
        self.num_candidates = num_candidates
        self.num_elites = num_elites
        self.planning_horizon = planning_horizon

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
                torch.ones((self.planning_horizon, self.posterior_model.action_dim),device=self.device),
            )

            # iteratively improve action distribution with CEM
            for _ in range(self.num_iterations):
                # sample action candidates
                # reshape to (planning_horizon, num_candidates, action_dim) for parallel exploration
                action_candidates = action_dist.sample([self.num_candidates])
                action_candidates = einops.rearrange(action_candidates, "n h a -> h n a")

                state = state_posterior.sample([self.num_candidates]).squeeze(-2)
                total_predicted_reward = torch.zeros(self.num_candidates, device=self.device)
                observation_trajectories = torch.zeros(
                    (self.planning_horizon, self.num_candidates, obs.shape[1]),
                    device=self.device,
                )

                # start generating trajectories starting from s_t using transition model
                for t in range(self.planning_horizon):
                    observation_trajectories[t] = self.observation_model(state=state)
                    total_predicted_reward += self.reward_model(state=state).squeeze()
                    # get next state from our prior (transition model)
                    next_state_prior = self.transition_model(
                        prev_state=state,
                        prev_action=torch.tanh(action_candidates[t]),
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
            
            # return only mean of the first action (MPC)
            actions = torch.tanh(mean)
            best_trajectory = observation_trajectories[:, elite_indexes, :].mean(dim=1)

        return actions.cpu().numpy(), best_trajectory.cpu().numpy()
    
    def reset(self):
        self.rnn_hidden = torch.zeros(1, self.posterior_model.rnn_hidden_dim, device=self.device)


class RSAgent:
    """
        action planning by Random Shooting
    """
    def __init__(
        self,
        transition_model,
        posterior_model,
        reward_model,
        observation_model,
        planning_horizon: int,
        num_candidates: int,
    ):
        self.transition_model = transition_model
        self.posterior_model = posterior_model
        self.reward_model = reward_model
        self.observation_model = observation_model
        self.num_candidates = num_candidates
        self.planning_horizon = planning_horizon

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

            action_dist = Uniform(
                low=-torch.ones((self.planning_horizon, self.posterior_model.action_dim), device=self.device),
                high=torch.ones((self.planning_horizon, self.posterior_model.action_dim),device=self.device),
            )

            action_candidates = action_dist.sample([self.num_candidates])
            action_candidates = einops.rearrange(action_candidates, "n h a -> h n a")

            state = state_posterior.sample([self.num_candidates]).squeeze(-2)
            total_predicted_reward = torch.zeros(self.num_candidates, device=self.device)
            observation_trajectories = torch.zeros(
                (self.planning_horizon, self.num_candidates, obs.shape[1]),
                device=self.device,
            )

            # start generating trajectories starting from s_t using transition model
            for t in range(self.planning_horizon):
                observation_trajectories[t] = self.observation_model(state=state)
                total_predicted_reward += self.reward_model(state=state).squeeze()
                # get next state from our prior (transition model)
                next_state_prior = self.transition_model(
                    prev_state=state,
                    prev_action=action_candidates[t],
                )
                state = next_state_prior.sample()

            # find the best action sequence
            max_index = total_predicted_reward.argmax()
            actions = action_candidates[:, max_index, :]
            best_trajectory = observation_trajectories[:, max_index, :]

        return actions.cpu().numpy(), best_trajectory.cpu().numpy()
    
    def reset(self):
        self.rnn_hidden = torch.zeros(1, self.posterior_model.rnn_hidden_dim, device=self.device)