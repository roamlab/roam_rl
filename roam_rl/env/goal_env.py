import gym


class GoalEnv(gym.GoalEnv):
    """ base class for a configurable gym environment

    users of this class must implement the following methods

    step
    reset
    render
    close
    seed
    compute_reward

    """

    def __init__(self, config_data, section_name):
        pass

    def step(self, action):
        super().step(action)

    def reset(self):
        super().reset()

    def render(self, mode='human'):
        super().render(mode=mode)

    def close(self):
        super().close()

    def seed(self, seed=None):
        super().seed(seed=seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        super().compute_reward(achieved_goal=achieved_goal, desired_goal=desired_goal, info=info)