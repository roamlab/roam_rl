from roam_rl.utils.factory import make
import gym


def make_robot_env(config_data, section_name):
    # create
    # 1. env classes for which the entry point is specified in config
    # 2. openai envs that are registered with gym

    base_type = config_data.get(section_name, 'type')
    if base_type == 'gym_registered':
        id = config_data.get(section_name, 'id')
        env = gym.make(id)
        if config_data.has_option(section_name, 'reward_type'):
            if hasattr(env.env, 'reward_type'):
                reward_type = config_data.get(section_name, 'reward_type')
                if reward_type == 'sparse' or reward_type == 'dense':
                    env.env.reward_type = reward_type
                else:
                    raise ValueError('reward type unknown')
            else:
                raise ValueError('reward_type cannot be configured for {}'.format(env.env))
        return env
    else:
        return make(config_data, section_name)

