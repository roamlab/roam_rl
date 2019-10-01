from baselines.ppo_her import ppo_her
from roamrl.openai_baselines.ppo import PPO


class PPOHER(PPO):

    def __init__(self, config_data, section_name):
        super().__init__(config_data, section_name)
        self._learn = ppo_her.learn

    def _get_parameter_descr_dict(self):
        parameters = super()._get_parameter_descr_dict()
        parameters = {
            **parameters,
            'buffer_age': 'int',
            'exp_ratio': 'float',
            'hs_strategy': 'str'
        }
        return parameters
