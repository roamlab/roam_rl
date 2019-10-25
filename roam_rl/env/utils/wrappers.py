import gym.wrappers
import ast

class FilterObservation(gym.wrappers.FilterObservation):

    def __init__(self, env, config_data, section_name):
        filter_keys = ast.literal_eval(config_data.get(section_name, 'filter_keys'))
        super().__init__(env, filter_keys=filter_keys)


class FlattenObservation(gym.wrappers.FlattenObservation):

    def __init__(self, env, config_data, section_name):
        super().__init__(env)