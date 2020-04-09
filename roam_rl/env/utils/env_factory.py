from confac import make
import gym


def make_env(config, section):
    """ create env using config

        envs specified by entrypoint can be created with make() but envs registered with gym
        need to be created differently ( _make_gym_registered_env())

    """
    assert (config.has_option(section, 'id') and config.has_option(section, 'entrypoint')) is False, \
        "cannot specify both id and entrypoint"

    if config.has_option(section, 'id'):
        env = _make_gym_registered_env(config, section)
    elif config.has_option(section, 'entrypoint'):
        env = make(config, section)
    else:
        raise ValueError('env unknown')

    return env


def _make_gym_registered_env(config, section):
    """ handles creation of envs registered with gym - including envs defined outside of gym as long as they are
    correctly registered with gym  """

    # If the env that is to be created is different module then include the name of the module to import as shown below
    # id: my_module:EnvName-v0, gym will 'import my_module' and then proceed to creating the env
    env_id = config.get(section, 'id')
    try:
        # Try creating the env with config, this will if the environment's __init__() either accepts config
        # and section as arguments directly or through **kwargs
        env = gym.make(id=env_id, config=config, section=section)
    except TypeError:
        # The above method will fail for all gym environments by OpenAI as their __init__() does not accept **kwargs,
        # so create the environment with just the id as the argument
        env = gym.make(id=env_id)

    # for gym's robotics environments (https://github.com/openai/gym/tree/master/gym/envs/robotics) the reward type
    # is configurable between the 'sparse'(default) and 'dense' reward.
    if config.has_option(section, 'reward_type'):
        if hasattr(env.env, 'reward_type'):
            reward_type = config.get(section, 'reward_type')
            if reward_type == 'sparse' or reward_type == 'dense':
                env.env.reward_type = reward_type
            else:
                raise ValueError('reward type unknown')
        else:
            raise ValueError('reward_type cannot be configured for {}'.format(env.env))

    return env




