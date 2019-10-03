from roam_rl.robot_env.random_sampler import RandomSampler
import numpy as np
import ast


class RectangularGoalSampler(RandomSampler):

    def __init__(self, config_data, section_name):
        super().__init__()
        self.x_range = np.asarray([float(x) for x in ast.literal_eval(config_data.get(section_name, 'x_range'))])
        self.y_range = np.asarray([float(x) for x in ast.literal_eval(config_data.get(section_name, 'y_range'))])
        self.center = np.asarray([float(x) for x in ast.literal_eval(config_data.get(section_name, 'center'))])

    def sample(self):
        rvals = self.rng.uniform(low=0, high=1, size=(2,))
        x = self.x_range[0] + rvals[0] * (self.x_range[1] - self.x_range[0])
        y = self.y_range[0] + rvals[1] * (self.y_range[1] - self.y_range[0])
        return np.asarray([x, y])
