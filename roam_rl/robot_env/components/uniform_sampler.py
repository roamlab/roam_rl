from roam_rl.robot_env.components.random_sampler import RandomSampler
import numpy as np
import ast


class UniformSampler(RandomSampler):

    def __init__(self, config, section):
        super().__init__()
        self.min = \
            np.asarray([float(x) for x in ast.literal_eval(config.get(section, 'min'))])
        self.max = \
            np.asarray([float(x) for x in ast.literal_eval(config.get(section, 'max'))])

    def sample(self):
        return self.rng.uniform(low=self.min, high=self.max)
