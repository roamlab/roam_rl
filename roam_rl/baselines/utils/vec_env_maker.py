from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_monitor import VecMonitor
from baselines.common.vec_env.vec_normalize import VecNormalize
from gym.utils.seeding import hash_seed
from copy import deepcopy


class VecEnvMaker(object):

    """ Callable class that takes instance of roam_learning.robot_env.EnvMaker and returns either a DummyVecEnv,
     SubprocVecEnv or ShmemVecEnv """

    def __init__(self, config, section):
        vectorize_type = config.get(section, 'type')
        if vectorize_type == 'dummy':
            self.vec_env_wrapper = DummyVecEnv
        elif vectorize_type == 'subproc':
            self.vec_env_wrapper = SubprocVecEnv
        elif vectorize_type == 'shmem':
            self.vec_env_wrapper = ShmemVecEnv
        else:
            raise ValueError('vectorize_type {} not recognized'.format(vectorize_type))
        self.nenvs = config.getint(section, 'nenvs')
        self.normalize_obs = config.getboolean(section, 'normalize_obs', fallback=False)
        self.normalize_ret = config.getboolean(section, 'normalize_ret', fallback=False)

    def __call__(self, env_maker, seed=None, monitor_file=None):
        """
        :param env_maker: instance of roam_learning.robot_env.EnvMaker
        :param seed: int that is used to generate seeds for vectorized envs
        :param monitor_file: path to a .csv file to log episode rewards, lengths etc,. of the vectorized envs
        :return: instance of either DummyVecEnv, SubprocVecEnv or ShmemVecEnv
        """
        # Create a list of env makers
        if seed is not None:
            assert isinstance(seed, int)
        env_makers = []
        for i in range(self.nenvs):
            env_makers += [deepcopy(env_maker)]
            if seed is not None:
                seed = hash_seed(seed)
                env_makers[i].set_seed(seed + i)

        # Create the vectorized envs
        envs = self.vec_env_wrapper(env_makers)

        # Monitor the envs before normalization
        if monitor_file is not None:
            envs = VecMonitor(envs, filename=monitor_file)
        if self.normalize_obs or self.normalize_ret:
            envs = VecNormalize(envs, ob=self.normalize_obs, ret=self.normalize_ret, use_tf=True)
        return envs
