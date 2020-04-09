import gym


class Env(gym.Env):

    """ base class for a configurable gym environment

    users of this class must implement the following methods

    step
    reset
    render
    close
    seed

    and also set these attributes

    action_space: The Space object corresponding to valid actions
    observation_space: The Space object corresponding to valid observations
    reward_range: A tuple corresponding to the min and max possible rewards

    For more details, refer https://github.com/openai/gym/blob/master/gym/core.py

    """

    def __init__(self, config, section):
        pass

    def step(self, action):
        """ Accepts an action and returns a tuple (observation, reward, done, info) """
        super().step(action)

    def reset(self):
        """ Resets the state of the environment and returns an initial observation. """
        super().reset()

    def render(self, mode='human'):
        """ Renders the environment. """
        super().render(mode=mode)

    def close(self):
        """ Overide this if your env needs garbage collection and cleanup """
        super().close()

    def seed(self, seed=None):
        """ Sets the seed for this env's random number generator(s). """
        super().seed(seed=seed)






