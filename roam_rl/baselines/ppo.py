import os
from confac import make
from baselines.common import set_global_seeds
from baselines.ppo2 import ppo2
from baselines import logger
from roam_rl.baselines.utils import VecEnvMaker
from roam_rl.baselines.models import get_network
from gym import spaces
import numpy as np


class PPO:

    """  Wrapper for baselines.ppo2 """

    def __init__(self, config, section):
        self._learn = ppo2.learn
        self.experiment_dir = None
        self.config = config
        self.section = section

        # ppo parameters
        params = self._get_parameter_descr_dict()
        params = config.get_section(section, params)

        # build and set network function to be passed to learn
        network_section = params['network']
        params['network'] = get_network(config, network_section)
        self.params = params

        # env
        env_maker_section = config.get(section, 'env_maker')
        self.env_maker = make(config, env_maker_section)
        vec_env_maker_section = config.get(section, 'vec_env_maker')
        self.vec_env_maker = VecEnvMaker(config, vec_env_maker_section)

        self.seed = config.getint(section, 'seed')

    def _get_parameter_descr_dict(self):

        """
        Returns a dictionary of parameter names and their type.
        These parameters will be obtained from config

        """
        parameters = {
            'nsteps': 'int',
            'ent_coef': 'float',
            'lr': 'eval',  # lambda fn can also be used to specify varying learning rate
            'vf_coef': 'float',
            'max_grad_norm': 'float',
            'gamma': 'float',
            'lam': 'float',
            'nminibatches': 'int',
            'noptepochs': 'int',
            'cliprange': 'float',
            'network': 'str',
            'value_network': 'str',
            'log_interval': 'int',
            'save_interval': 'int',
            'total_timesteps': 'int'  # to read int from int sci notation
        }

        return parameters

    def learn(self, model_path=None):

        # Create vec env
        set_global_seeds(self.seed)
        logdir = self.get_log_dir(self.experiment_dir, self.seed)   # setup ppo logging
        logger.configure(dir=logdir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])
        monitor_file_path = os.path.join(logdir, 'monitor.csv')
        env = self.vec_env_maker(self.env_maker, self.seed, monitor_file=monitor_file_path)

        # Learn
        # pylint: disable=E1125
        model = self._learn(env=env, **self.params, seed=self.seed, load_path=model_path)   # learn model

        # Save
        model.save(self.get_model_path(self.experiment_dir, self.seed))
        env.close()

    def set_experiment_dir(self, dir_name):
        self.env_maker.set_experiment_dir(dir_name)
        self.experiment_dir = dir_name

    def load(self, model_seed, model_checkpoint=None, env_seed=0, monitor_file=None):
        """ load a trained model from model_path
            supply config to modify env behaviour
        """
        env = self.vec_env_maker(self.env_maker, seed=env_seed, monitor_file=monitor_file)

        # train for 0 timesteps to load
        self.params['total_timesteps'] = 0
        model_path = self.get_model_path(self.experiment_dir, model_seed, model_checkpoint)
        # pylint: disable=E1125
        model = self._learn(env=env, **self.params, load_path=model_path)
        return model, env

    def run(self, model, env, stochastic=False):
        """ """
        obs = env.reset()
        _states = None
        # after training stochasticity of the policy is not relevant, 
        # set the actions to be mean of the policy
        if not stochastic:
            model.act_model.action = model.act_model.pi 

        def determinstic_action(pi):
            if isinstance(env.action_space, spaces.Box):
                return pi
            if isinstance(env.action_space, spaces.Discrete):
                return np.argmax(pi)
            if isinstance(env.action_space, spaces.MultiDiscrete):
                nvec = env.action_space.nvec
                action = np.zeros(len(nvec))
                start = 0
                for i in range(len(nvec)):
                    action[i] = np.argmax(pi[0][start:start+nvec[i]])
                    start += nvec[i]
                return action

        while True:
            action, _, _states, _ = model.step(obs)
            if not stochastic:
                # find action from pi
                action = determinstic_action(action)

            # pylint: disable=W0612
            obs, rewards, dones, info = env.step(action)
            env.render()

    @staticmethod
    def get_experiment_dir(path_to_experiments, robot_name, experiment_no):
        experiment_dir = os.path.join(path_to_experiments, 'experiments', robot_name, 'ppo', 'experiment_'\
                                      + str(experiment_no).zfill(2))
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir

    @staticmethod
    def get_model_path(experiments_dir, seed, model_checkpoint=None):
        model_dir = os.path.join(experiments_dir, 'seed-{}'.format(seed))
        os.makedirs(model_dir, exist_ok=True)
        if model_checkpoint:
            model_path = os.path.join(model_dir, 'checkpoints/{}'.format(str(model_checkpoint).zfill(5)))
        else:
            model_path = os.path.join(model_dir, 'model_{}.pkl'.format(seed))
        return model_path

    @staticmethod
    def get_log_dir(experiments_dir, seed):
        logdir = os.path.join(experiments_dir, 'seed-{}'.format(seed))
        os.makedirs(logdir, exist_ok=True)
        return logdir

    @staticmethod
    def get_config_path(load_dir, experiment_no):
        return os.path.join(load_dir, 'config_' + str(experiment_no).zfill(2) + '.cfg')
