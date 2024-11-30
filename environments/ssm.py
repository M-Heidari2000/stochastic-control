import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Callable

class NonLinearSSM(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
    }

    def __init__(
        self,
        A,
        b,
        B,
        C,
        f: Callable,
        Q,
        R,
        Ni,
        Ns=None,
        Nz=None,
        No=None,
        action_lo: float=-1.0,
        action_hi: float=1.0,
        render_mode: str=None,
        horizon: int= 1000,
    ):
        # Verify parameters' shapes
        assert A.shape[0] == A.shape[1]
        self.state_dim = A.shape[0]
        self.A = A.astype(np.float32)

        assert b.shape[0] == self.state_dim
        # Convert to column vector
        self.b = b.astype(np.float32).reshape(-1, 1)

        assert B.shape[0] == self.state_dim
        self.action_dim = B.shape[1]
        self.B = B.astype(np.float32)
        
        assert C.shape[1] == self.state_dim
        self.intermediate_state_dim = C.shape[0]
        self.C = C.astype(np.float32)
        
        dummy_input = np.zeros((self.intermediate_state_dim, 1))
        dummy_output = f(dummy_input)
        self.observation_dim = dummy_output.shape[0]
        self.f = f

        assert Q.shape == (self.state_dim, self.state_dim)
        self.Q = Q.astype(np.float32)
        assert R.shape == (self.action_dim, self.action_dim)
        self.R = R.astype(np.float32)

        assert Ni.shape == (self.state_dim, self.state_dim)
        self.Ni = Ni.astype(np.float32)

        self.Ns = Ns
        self.Nz = Nz
        self.No = No
        if Ns is not None:
            assert Ns.shape == (self.state_dim, self.state_dim)
            self.Ns = Ns.astype(np.float32)
        if Nz is not None:
            assert Nz.shape == (self.intermediate_state_dim, self.intermediate_state_dim)
            self.Nz = Nz.astype(np.float32)
        if No is not None:
            assert No.shape == (self.observation_dim, self.observation_dim)
            self.No = No.astype(np.float32)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.horizon = horizon

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim, ),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=action_lo,
            high=action_hi,
            shape=(self.action_dim, ),
            dtype=np.float32,
        )

    def _get_obs(self):
        z = self.C @ self._state
        if self.Nz is not None:
            nz = self.np_random.multivariate_normal(
                mean=np.zeros((self.intermediate_state_dim, )),
                cov=self.Nz,
            ).astype(np.float32).reshape(-1, 1)
            z = z + nz
            
        obs = self.f(z)
        if self.No is not None:
            no = self.np_random.multivariate_normal(
                mean=np.zeros((self.observation_dim, )),
                cov=self.No,
            ).astype(np.float32).reshape(-1, 1)
            obs = obs + no
        return obs

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        
        super().reset(seed=seed)
        options = options or {}
        initial_state = options.get("initial_state")

        if initial_state is not None:
            assert initial_state.shape == (self.state_dim, )
            self._state = initial_state.astype(np.float32).reshape(-1, 1)
        else:
            self._state = self.np_random.multivariate_normal(
                mean=np.zeros((self.state_dim, )),
                cov=self.Ni,
            ).astype(np.float32).reshape(-1, 1)
        
        self._step = 1
        observation = self._get_obs().flatten()
        info = {"state": self._state.copy().flatten()}

        return observation, info
            
    def step(
        self,
        action
    ):
        assert action.shape == self.action_space.shape
        action = action.astype(np.float32).reshape(-1, 1)

        # Calculate reward for current state and action
        reward = -(self._state.T @ self.Q @ self._state) - (action.T @ self.R @ action)

        # Calculate Next step
        self._state = self.A @ self._state + self.b + self.B @ action
        if self.Ns is not None:
            ns = self.np_random.multivariate_normal(
                mean=np.zeros((self.state_dim, )),
                cov=self.Ns,
            ).astype(np.float32).reshape(-1, 1)
            self._state = self._state + ns
        self._step += 1
        
        info = {"state": self._state.copy().flatten()}
        truncated = bool(self._step >= self.horizon)
        # In this env, episodes never terminate
        terminated = False
        reward = reward.item()
        obs = self._get_obs().flatten()

        return obs, reward, terminated, truncated, info 

    def render(self):
        pass