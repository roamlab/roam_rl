from roam_rl.robot_env.components import RewardFunc


class PointRobotReward(RewardFunc):
    """ make the point robot go to -1.0 on the x-axis """

    def __call__(self, obs, action):
        x = obs['state'][0, 0]
        reward = -(x-1.0)**2
        return reward
