class ObservationFunc(object):

    def __init__(self, config_data, section_name):
        pass

    def __call__(self, obs):
        env_obs = self.get_obs(obs)
        for key in env_obs.keys():
            env_obs[key] = env_obs[key].reshape(-1)
        return env_obs

    def get_obs(self, obs):
        return obs
