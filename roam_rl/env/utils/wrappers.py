import gym.wrappers
import ast

class FilterObservation(gym.wrappers.FilterObservation):

    def __init__(self, env, config, section):
        filter_keys = ast.literal_eval(config.get(section, 'filter_keys'))
        super().__init__(env, filter_keys=filter_keys)


class FlattenObservation(gym.wrappers.FlattenObservation):

    def __init__(self, env, config, section):
        super().__init__(env)
