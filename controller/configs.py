from dataclasses import dataclass, asdict
from typing import Optional, Dict


@dataclass
class TrainConfig:
    seed: int = 0
    log_dir: str = 'log'
    test_interval: int = 10
    action_repeat: int = 1
    state_dim: int = 30
    rnn_hidden_dim: int = 200
    rnn_input_dim: int = 200
    min_std: float = 1e-2
    buffer_capacity: int = 1000000
    all_episodes: int = 1000
    seed_episodes: int = 5
    collect_interval: int = 100
    batch_size: int = 50
    chunk_length: int = 50
    lr: float = 1e-3
    eps: float = 1e-5
    clip_grad_norm: int = 1000
    free_nats: int = 3
    kl_beta: float = 1
    planning_horizon: int = 12
    num_iterations: int = 10
    num_candidates: int = 1000
    num_elites: int = 100
    action_noise_var: float = 0.3

    dict = asdict


@dataclass
class TestConfig:
    model_dir: str
    action_repeat: int = 4
    episodes: int = 1
    planning_horizon: int = 12
    num_iterations: int = 10
    num_candidates: int = 1000
    num_top_candidates: int = 100

    dict = asdict