import gym
from gym.utils import seeding
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import ast
from warnings import warn
from roam_utils.factory import make
from roam_rl.utils.path_generator import PathGenerator


class RobotGoalEnv(gym.GoalEnv):

    """ goal based env class for robot world
        - robot_world which can either be simulated or real
        - state_sampler for sampling the initial state of the robot world at the start of an episode
          state does not correspond to full state of the robot and can be probably named something better
        - compute_reward is an object of the callable reward function class, returns reward as a function of observation
          and action
        - get_obs is also an object of callable class for observation function, converts the robot_world observation to
          observation as seen by the RL agent
        - goal sampler samples new goal for each episode
    """

    def __init__(self, config_data, section_name):
        self.max_episode_steps = config_data.getint(section_name, 'max_episode_steps')
        robot_world_section_name = config_data.get(section_name, 'robot_world')
        self.robot_world = make(config_data, robot_world_section_name)
        state_sampler_section_name = config_data.get(section_name, 'state_sampler')
        self.state_sampler = make(config_data, state_sampler_section_name)
        reward_func_section_name = config_data.get(section_name, 'reward_func')
        self.reward_func = make(config_data, reward_func_section_name)
        obs_func_section_name = config_data.get(section_name, 'observation_func')
        self.get_obs = make(config_data, obs_func_section_name)
        goal_sampler_section_name = config_data.get(section_name, 'goal_sampler')
        self.goal_sampler = make(config_data, goal_sampler_section_name)
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
        if config_data.has_option(section_name, 'render_gui'):
            render_gui_section_name = config_data.get(section_name, 'render_gui')
            self.render_gui = make(config_data, render_gui_section_name)
            self.render_gui.add_subject(self.robot_world)

    def step(self, action):
        self.robot_world.take_action(action.reshape(-1, 1))
        obs = self.robot_world.get_obs()
        env_obs = self.get_obs(obs)
        env_obs['desired_goal'] = deepcopy(self.goal)
        assert 'achieved_goal' in env_obs.keys(), 'env_obs does not contain "achieved goal", ensure observation' \
                                                 'function is correctly implemented'
        reward = self.compute_reward(achieved_goal=env_obs['achieved_goal'], desired_goal=env_obs['desired_goal'],
                                     info={'action':action})
        self.steps += 1
        done = False
        if self.steps >= self.max_episode_steps:
            done = True
        info = {}
        return env_obs, reward, done, info

    def reset(self):
        state = self.state_sampler.sample()
        self.robot_world.reset_time()
        self.robot_world.set_state(state.reshape(-1, 1))
        self.goal = self.goal_sampler.sample()
        obs = self.robot_world.get_obs()
        env_obs = self.get_obs(obs)
        env_obs['desired_goal'] = deepcopy(self.goal)
        self.steps = 0
        if self.render_gui is not None:
            self.render_gui.reset()
            self.render_gui.set_goal(self.goal)
        return env_obs

    def render(self, mode='human'):
        if self.render_gui is not None:
            self.render_gui.render()
            if hasattr(self.render_gui, 'record_sim'):
                if self.render_gui.record_sim is True:
                    save_path = PathGenerator.get_gui_render_path(self.render_gui.render_dir,
                                                                  self.render_gui.frame_count)
                    self.render_gui.save_frame_based_on_fps(save_path)
        else:
            warn('render() called without initializing render_gui for robot_world')

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.state_sampler.set_rng(self.np_random)
        self.goal_sampler.set_rng(self.np_random)
        return [seed]

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.reward_func(achieved_goal=achieved_goal, desired_goal=desired_goal, info=info)

