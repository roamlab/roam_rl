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
