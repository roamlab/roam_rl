import os
from baselines.common import set_global_seeds
from baselines.ppo2 import ppo2
from baselines import logger
from baselines.common.vec_env.vec_normalize import VecNormalize
from roam_rl.utils import PathGenerator
from roam_utils.provenance.config_helpers import pull_from_config
from roam_rl.openai_baselines.utils import VecEnvMaker
from roam_utils.factory import make
from roam_rl.openai_baselines.models import get_network


class PPO(object):

    """  Wrapper for ppo2 from OpenAI baselines """

    def __init__(self, config_data, section_name):
        self._learn = ppo2.learn
        self.experiment_dir = None
        self.config_data = config_data
        self.section_name = section_name

        # ppo parameters
        params = self._get_parameter_descr_dict()               
        params = pull_from_config(params, config_data, section_name)

        # build and set network function to be passed to learn
        network_section_name = params['network']                
        params['network'] = get_network(config_data, network_section_name)
        self.params = params

        # env
        env_maker_section_name = config_data.get(section_name, 'env_maker')
        self.env_maker = make(config_data, env_maker_section_name)
        vec_env_maker_section_name = config_data.get(section_name, 'vec_env_maker')
        self.vec_env_maker = VecEnvMaker(config_data, vec_env_maker_section_name)

        self.seed = config_data.getint(section_name, 'seed')

    def _get_parameter_descr_dict(self):
        
        """ Returns a dictionary of parameter names and their type. 
        These parameters will be obtained from config  """
        
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
            'total_timesteps': 'float2int'  # to read int from int sci notation
        }

        return parameters

    def learn(self, model_path=None):

        # Create vec env
        set_global_seeds(self.seed)
        logdir = PathGenerator.get_ppo_log_dir(self.experiment_dir, self.seed)   # setup ppo logging
        logger.configure(dir=logdir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])
        monitor_file_path = os.path.join(logdir, 'monitor.csv')
        env = self.vec_env_maker(self.env_maker, self.seed, monitor_file=monitor_file_path)

        # Learn
        # pylint: disable=E1125
        model = self._learn(env=env, **self.params, seed=self.seed, load_path=model_path)   # learn model

        # Save
        model.save(PathGenerator.get_ppo_model_path(self.experiment_dir, self.seed))
        env.close()

    def set_experiment_dir(self, dir_name):
        self.env_maker.set_experiment_dir(dir_name)
        self.experiment_dir = dir_name

    def load(self, model_seed, model_checkpoint=None, env_seed=0, monitor_file=None):
        """ load a trained model from model_path
            supply config_data to modify env behaviour
        """
        env = self.vec_env_maker(self.env_maker, seed=env_seed, monitor_file=monitor_file)

        # train for 0 timesteps to load
        self.params['total_timesteps'] = 0
        model_path = PathGenerator.get_ppo_model_path(self.experiment_dir, model_seed, model_checkpoint)
        # pylint: disable=E1125
        model = self._learn(env=env, **self.params, load_path=model_path)
        return model, env

    def run(self, model, env):
        """ """
        obs = env.reset()
        _states = None
        # after training stochasticity of the policy is not relevant, set the actions to be mean of the policy
        model.act_model.action = model.act_model.pi
        while True:
            action, _, _states, _ = model.step(obs)
            # pylint: disable=W0612
            obs, rewards, dones, info = env.step(action)
            env.render()
