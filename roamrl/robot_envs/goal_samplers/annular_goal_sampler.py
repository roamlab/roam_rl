from roamrl.utils.random_sampler import RandomSampler
import numpy as np
import ast
from math import pi, cos, sin


class AnnularGoalSampler(RandomSampler):

    def initialize_from_config(self, config_data, section_name):
        super().__init__()
        self.radius_range = \
            np.asarray([float(x) for x in ast.literal_eval(config_data.get(section_name, 'radius_range'))])
        self.center = np.asarray([float(x) for x in ast.literal_eval(config_data.get(section_name, 'center'))])

    def sample(self):
        rvals = self.rng.uniform(low=0, high=1, size=(2,))
        radius = self.radius_range[0] + rvals[0] * (self.radius_range[1] - self.radius_range[0])
        ang = rvals[1] * 2 * pi
        return np.asarray([radius*cos(ang) + self.center[0], radius*sin(ang) + self.center[1]])
