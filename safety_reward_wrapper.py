import gymnasium as gym
import numpy as np

class SafetyRewardWrapper(gym.Wrapper):
    """
    Reward shaping wrapper for highway-v0 with Kinematics observations.

    Observation format assumed:
      obs[i] = [presence, x, y, vx, vy]
    with obs[0] = ego vehicle and obs[1:] = nearby vehicles

    Uses normalised coordinates/distances from highway-env config
    """

    def __init__(
        self,
        env,
        collision_penalty=2.0,
        headway_threshold=0.12,
        headway_penalty=1.5,
        lane_change_penalty=0.02,
        high_speed_threshold=0.32,
        close_distance_threshold=0.10,
        high_speed_close_penalty=0.5,
    ):
        super().__init__(env)
        self.collision_penalty = collision_penalty
        self.headway_threshold = headway_threshold
        self.headway_penalty = headway_penalty
        self.lane_change_penalty = lane_change_penalty
        self.high_speed_threshold = high_speed_threshold
        self.close_distance_threshold = close_distance_threshold
        self.high_speed_close_penalty = high_speed_close_penalty

        self.prev_lane = None
        self.last_info = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_lane = self._get_lane_id()
        self.last_info = {
            "safety_penalty": 0.0,
            "front_distance": None,
            "front_rel_speed": None,
            "lane_changed": False,
            "collision": False,
        }
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        penalty = 0.0
        lane_changed = False
        collision = bool(terminated)

        # Collision penalty
        if collision:
            penalty += self.collision_penalty

        # Front vehicle risk penalty
        front_distance, front_rel_speed = self._front_vehicle_metrics(obs)
        if front_distance is not None and front_distance < self.headway_threshold:
            penalty += self.headway_penalty * (self.headway_threshold - front_distance)

        # Lane change penalty
        lane = self._get_lane_id()
        if self.prev_lane is not None and lane is not None and lane != self.prev_lane:
            penalty += self.lane_change_penalty
            lane_changed = True
        self.prev_lane = lane

        # Fast and close penalty
        ego_vx = float(obs[0, 3])
        if front_distance is not None:
            if ego_vx > self.high_speed_threshold and front_distance < self.close_distance_threshold:
                penalty += self.high_speed_close_penalty

        shaped_reward = reward - penalty

        safety_info = {
            "base_reward": float(reward),
            "safety_penalty": float(penalty),
            "front_distance": None if front_distance is None else float(front_distance),
            "front_rel_speed": None if front_rel_speed is None else float(front_rel_speed),
            "lane_changed": lane_changed,
            "collision": collision,
            "shaped_reward": float(shaped_reward),
        }

        info = dict(info)
        info.update(safety_info)
        self.last_info = safety_info

        return obs, shaped_reward, terminated, truncated, info

    def _get_lane_id(self):
        """
        Extract a stable lane identifier from the underlying env vehicle.
        """
        vehicle = getattr(self.env.unwrapped, "vehicle", None)
        if vehicle is None:
            return None

        lane_index = getattr(vehicle, "lane_index", None)
        if lane_index is None:
            return None

        # highway-env uses tuples like (road_from, road_to, lane_id)
        if isinstance(lane_index, tuple) and len(lane_index) >= 3:
            return lane_index[-1]

        return lane_index

    def _front_vehicle_metrics(self, obs):
        """
        Find the nearest vehicle approximately in front of ego and in same lane.
        Uses normalized kinematics observation:
          obs[:, 0]=presence, obs[:, 1]=x, obs[:, 2]=y, obs[:, 3]=vx, obs[:, 4]=vy
        """
        ego = obs[0]
        ego_x = float(ego[1])
        ego_y = float(ego[2])
        ego_vx = float(ego[3])

        best_dist = None
        best_rel_speed = None

        for i in range(1, obs.shape[0]):
            if obs[i, 0] < 0.5:
                continue

            veh_x = float(obs[i, 1])
            veh_y = float(obs[i, 2])
            veh_vx = float(obs[i, 3])

            dx = veh_x - ego_x
            dy = abs(veh_y - ego_y)

            # in front and roughly same lane
            if dx > 0 and dy < 0.06:
                if best_dist is None or dx < best_dist:
                    best_dist = dx
                    best_rel_speed = ego_vx - veh_vx

        return best_dist, best_rel_speed