# shared_core_config.py
# Shared benchmark configuration for the core task.
# Environment : highway-v0
# Observations: Kinematics (5 vehicles x 5 features = 25-dim flat vector)
# Actions     : DiscreteMetaAction (5 discrete actions)

HIGHWAY_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": True,
        "absolute": False,
        "order": "sorted",
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 40,
    "initial_spacing": 2,
    "collision_reward": -1.0,
    "reward_speed_range": [20, 30],
    "simulation_frequency": 15,
    "policy_frequency": 1,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "offscreen_rendering": True,
}

# Our DQN hyperparameters
DQN_HYPERPARAMS = {
    "gamma": 0.9,
    "batch_size": 64,
    "buffer_capacity": 15000,
    "update_target_every": 200,
    "epsilon_start": 1.0,
    "decrease_epsilon_factor": 200,
    "epsilon_min": 0.05,
    "learning_rate": 5e-4,
    "hidden_size": 256,
    "n_episodes": 300,
}

# Stable-Baselines3 hyperparameters (matched to our DQN for fair comparison)
SB3_HYPERPARAMS = {
    "gamma": 0.9,
    "learning_rate": 5e-4,
    "batch_size": 64,
    "buffer_size": 15000,
    "learning_starts": 200,
    "target_update_interval": 200,
    "train_freq": 1,
    "exploration_fraction": 0.3,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "policy_kwargs": {"net_arch": [256, 256]},
    "total_timesteps": 15000,
}
