from roamrl.utils.random_sampler import RandomSampler
import numpy as np
import ast


class FixedGoalSampler(RandomSampler):

    def __init__(self, config_data, section_name):
        super().__init__()
        self.goal = np.asarray([float(x) for x in ast.literal_eval(config_data.get(section_name, 'goal'))])

    def sample(self):
        return self.goal.copy()