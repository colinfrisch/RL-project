from copy import deepcopy
from shared_core_config import HIGHWAY_CONFIG, DQN_HYPERPARAMS

LAGRANGIAN_HIGHWAY_CONFIG = deepcopy(HIGHWAY_CONFIG)

LAGRANGIAN_HYPERPARAMS = deepcopy(DQN_HYPERPARAMS)
LAGRANGIAN_HYPERPARAMS["n_episodes"] = 2000

# Hyperparamètres spécifiques au Lagrangian
LAGRANGIAN_HYPERPARAMS["collision_threshold"] = 0.05  # contrainte : max 5% de collisions
LAGRANGIAN_HYPERPARAMS["lambda_lr"] = 0.01            # vitesse d'ajustement de lambda
LAGRANGIAN_HYPERPARAMS["lambda_init"] = 0.0           # lambda démarre à 0
LAGRANGIAN_HYPERPARAMS["lambda_max"] = 10.0           # lambda ne peut pas dépasser 10
