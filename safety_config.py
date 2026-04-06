from copy import deepcopy
from shared_core_config import HIGHWAY_CONFIG, DQN_HYPERPARAMS, SB3_HYPERPARAMS

# Base environment config for safety training : same benchmark as core
SAFETY_HIGHWAY_CONFIG = deepcopy(HIGHWAY_CONFIG)

# Safety reward shaping hyperparameters
SAFETY_REWARD_CONFIG = {
    "collision_penalty": 2.0, # added on collision
    "headway_threshold": 0.12, # in normalised x-distance units
    "headway_penalty": 1.5, # scales deficit under threshold
    "lane_change_penalty": 0.02, # penalty each time lane changes
    "high_speed_threshold": 0.32, # normalised vx
    "close_distance_threshold": 0.10, # normalised x-distance
    "high_speed_close_penalty": 0.5, # risky fast-close behavior
}

SAFETY_DQN_HYPERPARAMS = deepcopy(DQN_HYPERPARAMS)
SAFETY_SB3_HYPERPARAMS = deepcopy(SB3_HYPERPARAMS)

# Dense and hard eval configs
# SAFETY_EVAL_DENSE_CONFIG = deepcopy(HIGHWAY_CONFIG)
# SAFETY_EVAL_DENSE_CONFIG["vehicles_count"] = 70
# SAFETY_EVAL_DENSE_CONFIG["initial_spacing"] = 1.5

# SAFETY_EVAL_HARD_CONFIG = deepcopy(HIGHWAY_CONFIG)
# SAFETY_EVAL_HARD_CONFIG["vehicles_count"] = 90
# SAFETY_EVAL_HARD_CONFIG["initial_spacing"] = 1.0
# SAFETY_EVAL_HARD_CONFIG["duration"] = 60