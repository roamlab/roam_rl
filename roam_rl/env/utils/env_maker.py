import ast
from gym.utils import seeding
from roam_rl.env.utils import make_env
from roam_rl.utils.path_generator import PathGenerator
from roam_utils import factory
import warnings


class EnvMaker(object):
    """ callable class for creating env
    __call__ method creates an env and sets seed for the env if one has been configured
    """

    def __init__(self, config_data, section_name):
        self.seed = None
        self.experiment_dir = None
        self.config_data = config_data
        self.section_name = section_name

    def set_seed(self, seed):
        assert isinstance(seed, int)
        self.seed = seeding.hash_seed(seed)

    def set_experiment_dir(self, experiment_dir):
        self.experiment_dir = experiment_dir

    def __call__(self):
        env_section_name = self.config_data.get(self.section_name, 'env')
        env = make_env(config_data=self.config_data, section_name=env_section_name)
        if type(self.seed) is int:
            env.seed(self.seed)
        else:
            warnings.warn("seed not set, using global RNG ")
        if self.experiment_dir is not None:
            try:
                env.render_gui.set_render_dir(PathGenerator.get_record_sim_dir(self.experiment_dir))
            except AttributeError:
                pass
        return env

    def __deepcopy__(self, memodict={}):
        env_maker = self.__class__(self.config_data, self.section_name)
        env_maker.seed = self.seed
        env_maker.experiment_dir = self.experiment_dir
        return env_maker


class WrappedEnvMaker(EnvMaker):

    """ env maker class that also wraps the base env with gym observation wrapper

    currently only FlattenDictWrapper and DictInputWrapper are supported and other wrappers can be included on
    need basis

    """

    def __init__(self, config_data, section_name):
        super().__init__(config_data, section_name)

    def __call__(self):
        env = super().__call__()
        config_data = self.config_data
        section_name = self.section_name
        if config_data.has_option(section_name, 'wrappers'):
            wrappers = ast.literal_eval(config_data.get(section_name, 'wrappers'))
            for wrapper_section_name in wrappers:
                wrapper = factory.get_attr(config_data, wrapper_section_name)
                env = wrapper(env, config_data, wrapper_section_name)
        return env















