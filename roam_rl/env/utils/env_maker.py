from gym.utils import seeding
from roam_rl.env.utils import make_env
import warnings
from confac import get, make


class EnvMaker(object):
    """ callable class for creating env
    __call__ method creates an env and sets seed for the env if one has been configured
    """

    def __init__(self, config, section):
        self.seed = None
        self.experiment_dir = None
        self.config = config
        self.section = section

    def set_seed(self, seed):
        assert isinstance(seed, int)
        self.seed = seeding.hash_seed(seed)

    def set_experiment_dir(self, experiment_dir):
        self.experiment_dir = experiment_dir

    def __call__(self):
        env_section = self.config.get(self.section, 'env')
        env = make_env(config=self.config, section=env_section)
        if type(self.seed) is int:
            env.seed(self.seed)
        else:
            warnings.warn("seed not set, using global RNG ")
        return env

    def __deepcopy__(self, memodict={}):
        env_maker = self.__class__(self.config, self.section)
        env_maker.seed = self.seed
        env_maker.experiment_dir = self.experiment_dir
        return env_maker


class WrappedEnvMaker(EnvMaker):

    """ env maker class that also wraps the base env with gym observation wrapper

    currently only FlattenDictWrapper and DictInputWrapper are supported and other wrappers can be included on
    need basis

    """

    def __init__(self, config, section):
        super().__init__(config, section)

    def __call__(self):
        env = super().__call__()
        config = self.config
        section = self.section
        if config.has_option(section, 'wrappers'):
            wrappers = config.getlist(section, 'wrappers')
            for wrapper_section in wrappers:
                wrapper = get(config, wrapper_section)
                env = wrapper(env, config, wrapper_section)
        return env

