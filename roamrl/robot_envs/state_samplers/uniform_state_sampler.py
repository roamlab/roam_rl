from roamrl.utils.random_sampler import RandomSampler
import numpy as np
import ast


class UniformStateSampler(RandomSampler):

    def __init__(self, config_data, section_name):
        super().__init__()
        self.state_min = \
            np.asarray([float(x) for x in ast.literal_eval(config_data.get(section_name, 'state_min'))])
        self.state_max = \
            np.asarray([float(x) for x in ast.literal_eval(config_data.get(section_name, 'state_max'))])

    def sample(self):
        return self.rng.uniform(low=self.state_min, high=self.state_max)