import ast
from gym.utils import seeding
from roamrl.robot_envs.robot_env_factory import make_robot_env
from roamrl.utils.path_generator import PathGenerator
from gym.wrappers import FlattenDictWrapper
import warnings
try:
    from gym.wrappers import DictInputWrapper
except ImportError:
    warnings.warn("Could not import DictInputWrapper from gym.wrappers")


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
        robot_env_section_name = self.config_data.get(self.section_name, 'robot_env')
        env = make_robot_env(config_data=self.config_data, section_name=robot_env_section_name)
        if type(self.seed) is int:
            env.seed(self.seed)
        else:
            warnings.warn("seed not set, using global RNG ")
        if self.experiment_dir is not None and hasattr(env, 'render_gui'):
            env.render_gui.set_render_dir(PathGenerator.get_record_sim_dir(self.experiment_dir))
        return env

    def __deepcopy__(self, memodict={}):
        env_maker = self.__class__()
        env_maker.initialize_from_config(self.config_data, self.section_name)
        env_maker.seed = self.seed
        env_maker.experiment_dir = self.experiment_dir
        return env_maker


class WrapObsEnvMaker(EnvMaker):

    """ env maker class that also wraps the base env with gym observation wrapper

    currently only FlattenDictWrapper and DictInputWrapper are supported and other wrappers can be included on
    need basis

    """

    def __init__(self, config_data, section_name):
        super().__init__(config_data, section_name)
        if config_data.has_option(section_name, 'observation_wrapper'):
            observation_wrapper = config_data.get(section_name, 'observation_wrapper')
            if observation_wrapper == "FlattenDictWrapper":
                self.observation_wrapper = FlattenDictWrapper
            elif observation_wrapper == "DictInputWrapper":
                self.observation_wrapper = DictInputWrapper
            self.observation_keys = ast.literal_eval(config_data.get(section_name, 'observation_keys'))

    def __call__(self):
        env = super().__call__()
        if self.observation_wrapper is not None:
            assert type(self.observation_keys) is list, 'observation keys must be a list'
            for obs_key in self.observation_keys:
                assert type(obs_key) is str, 'observation keys must be a list of strings'
            env = self.observation_wrapper(env, self.observation_keys)
        return env