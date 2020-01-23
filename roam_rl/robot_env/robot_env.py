import gym
from gym.utils import seeding
from collections import OrderedDict
import numpy as np
import ast
from warnings import warn
from roam_rl.env import Env
from roam_utils.factory import make
from roam_rl.utils.path_generator import PathGenerator


class RobotEnv(Env):

    """ env class for robot world

        - robot_world:  RobotWorld (ex. SimulatedRobotWorld) 
        - state_sampler for sampling the initial state of the robot world at the start of an episode
          state need not correspond to full state of the robot
          (state_sampler can be named something better)
        - compute_reward is an object of the callable RewardFunc, returns reward as a function of obs  and action
        - get_obs is also an object of callable class for observation function, converts the robot_world observation to
          observation as seen by the RL agent
    """

    def __init__(self, config_data, section_name):
        super().__init__(config_data=config_data, section_name=section_name)
        self.max_episode_steps = config_data.getint(section_name, 'max_episode_steps')
        robot_world_section_name = config_data.get(section_name, 'robot_world')
        self.robot_world = make(config_data, robot_world_section_name)
        state_sampler_section_name = config_data.get(section_name, 'state_sampler')
        self.state_sampler = make(config_data, state_sampler_section_name)
        reward_func_section_name = config_data.get(section_name, 'reward_func')
        self.compute_reward = make(config_data, reward_func_section_name)
        obs_func_section_name = config_data.get(section_name, 'observation_func')
        self.get_obs = make(config_data, obs_func_section_name)
        env_obs = self.reset()
        observation_spaces = OrderedDict()
        for key in env_obs.keys():
            observation_spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape=env_obs[key].shape, dtype='float32')
        self.observation_space = gym.spaces.Dict(observation_spaces)
        self.action_space_bounds = [-1.0, 1.0]
        if config_data.has_option(section_name, 'action_space_bounds'):
            self.action_space_bounds = [float(x) for x in ast.literal_eval(config_data.get(section_name,
                                                                                           'action_space_bounds'))]
        self.action_space = gym.spaces.Box(low=self.action_space_bounds[0], high=self.action_space_bounds[1],
                                           shape=(self.robot_world.get_action_dim(),))
        self.steps = 0
        self.render_gui = None
        if config_data.has_option(section_name, 'render_gui'):
            render_gui_section_name = config_data.get(section_name, 'render_gui')
            self.render_gui = make(config_data, render_gui_section_name)

        self.np_random = np.random

    def step(self, action):
        assert type(action) is np.ndarray
        self.robot_world.take_action(action.reshape(-1, 1))
        obs = self.robot_world.get_obs()
        reward = self.compute_reward(obs, action)
        self.steps += 1
        done = False
        if self.steps >= self.max_episode_steps:
            done = True
        info = {}
        return self.get_obs(obs), reward, done, info

    def reset(self):
        state = self.state_sampler.sample()
        self.robot_world.reset_time()
        self.robot_world.set_state(state.reshape(-1, 1))
        obs = self.robot_world.get_obs()
        self.steps = 0
        return self.get_obs(obs)

    def render(self, mode='human'):
        if self.render_gui is not None:
            self.render_gui.render()
        else:
            warn('render() called without initializing render_gui for robot_world')

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.state_sampler.set_rng(self.np_random)
        return [seed]



