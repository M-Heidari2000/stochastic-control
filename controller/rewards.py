class LQRReward:

    def __init__(self, Q, R, target_state):
        self.Q = Q
        self.R = R
        self.target_state = target_state.unsqueeze(0)

    
    def __call__(self, state, action):
        reward = -(state - self.target_state) @ self.Q @ (state - self.target_state).T
        reward -= action @ self.R @ action.T
        return reward.diag()