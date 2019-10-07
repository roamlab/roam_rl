from roam_rl.robot_env.components import RewardFunc
import numpy as np


class PointRobotReward(RewardFunc):

    def __call__(self, obs, action):
        x = obs['state'][0, 0]
        reward = -(x-1.0)**2
        return reward

