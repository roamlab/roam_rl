from roam_rl.utils.factory import make
import gym


def make_env(config_data, section_name):
    """ create env using config_data, envs registered with gym need a specific way to create the env,
     make_env() and make_gym_registered_env() exist as solution to this problem
    Cases:
    1. entry_point for env class is specified
    2. id by which the env is registered with gym is specified

    """
    base_type = None
    if config_data.has_option(section_name, 'type'):
        base_type = config_data.get(section_name, 'type')

    # entry_point for env class is specified
    if base_type is None:
        env = make(config_data, section_name)

    # id by which the env is registered with gym is specified
    elif base_type == 'gym_registered':
        env = _make_gym_registered_env(config_data, section_name)

    else:
        raise ValueError('env type {} unknown'.format(base_type))

    return env


def _make_gym_registered_env(config_data, section_name):
    """ handles creation of envs registered with gym - including envs defined outside of gym as long as they are
    correctly registered with gym  """

    # If the env that is to be created is different module then include the name of the module to import as shown below
    # id: my_module:EnvName-v0, gym will 'import my_module' and then proceed to creating the env
    env_id = config_data.get(section_name, 'id')
    try:
        # Try creating the env with config_data, this will if the environment's __init__() either accepts config_data
        # and section_name as arguments directly or through **kwargs
        env = gym.make(id=env_id, config_data=config_data, section_name=section_name)
    except TypeError:
        # The above method will fail for all gym environments by OpenAI as their __init__() does not accept **kwargs,
        # so create the environment with just the id as the argument
        env = gym.make(id=env_id)

    # for gym's robotics environments (https://github.com/openai/gym/tree/master/gym/envs/robotics) the reward type
    # is configurable between the 'sparse'(default) and 'dense' reward.
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




