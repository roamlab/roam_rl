import numpy as np


class RewardFunc(object):

    def __init__(self):
        pass

    def initialize_from_config(self, config_data, section_name):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError




