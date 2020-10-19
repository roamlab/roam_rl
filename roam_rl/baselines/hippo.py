from roam_rl.baselines.ppo import PPO
from hippo import hippo

class HIPPO(PPO):

    def __init__(self, config, section):
        super().__init__(config, section)
        self.reward_fn = hippo.extract_reward_fn(self.env_maker)
        # wrap hippo.learn to sneak-in the reward function
        def learn(*args, **kwargs):
            return hippo.learn(*args, **kwargs, reward_fn=self.reward_fn)
        self._learn = learn

    def _get_parameter_descr_dict(self):

        parameters =  super()._get_parameter_descr_dict()
        parameters.update(
            {
                'nbatch': 'int',
                'mode': 'str',
                'use_buffer': 'bool',
                'buffer_capacity': 'int',
                'hindsight': 'float'
            }
        )
        return parameters
