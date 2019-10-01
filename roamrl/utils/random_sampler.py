import numpy as np


class RandomSampler(object):

    def __init__(self, rng=None):
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random

    def set_rng(self, rng):
        if isinstance(rng, np.random.RandomState):
            self.rng = rng
        else:
            raise ValueError('rng {} must be instance of np.random.RandomState')

    def sample(self):
        raise NotImplementedError

    def get_shape(self):
        return self.sample().shape