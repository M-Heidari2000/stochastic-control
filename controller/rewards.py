import torch
import numpy as np


class LQRReward:

    def __init__(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        target_state: np.ndarray,
    ):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.Q = torch.as_tensor(Q, device=self.device)
        self.R = torch.as_tensor(R, device=self.device)
        self.target_state = torch.as_tensor(target_state, device=self.device).unsqueeze(0)

    
    def __call__(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        reward = -(state - self.target_state) @ self.Q @ (state - self.target_state).T
        reward -= action @ self.R @ action.T
        return reward.diag()