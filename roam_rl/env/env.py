import gym


class Env(gym.Env):

    """ base class for a configurable gym environment

    users of this class must implement the following methods

    step
    reset
    render
    close
    seed

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








