from roam_rl.robot_env.components import RewardFunc
import numpy as np


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class GoalRewardFunc(RewardFunc):

    def __call__(self, achieved_goal, desired_goal, action=None):
        raise NotImplementedError


class DenseGoalRewardFunc(GoalRewardFunc):

    def __init__(self, config, section):
        super().__init__(config, section)
        achieved_goal_reward = config.get(section, 'achieved_goal_reward')
        if achieved_goal_reward == 'none':
            self.compute_reward_achieved_goal = self.zero_fn(config, section)
        elif achieved_goal_reward == 'linear':
            self.compute_reward_achieved_goal = self.reward_for_achieved_goal__linear(config, section)
        elif achieved_goal_reward == 'quadratic':
            self.compute_reward_achieved_goal = self.reward_for_achieved_goal__quadratic(config, section)
        elif achieved_goal_reward == 'smooth_abs':
            self.compute_reward_achieved_goal = self.reward_for_achieved_goal__smooth_abs(config, section)
        else:
            raise ValueError('achieved_goal_reward_type {} not recognized'.format(achieved_goal_reward))

        action_reward_type = config.get(section, 'action_reward')
        if action_reward_type == 'none':
            self.compute_reward_action = self.zero_fn(config, section)
        elif action_reward_type == 'quadratic':
            self.compute_reward_action = self.reward_for_action_taken__quadratic(config, section)
        elif action_reward_type == 'action_limiting':
            self.compute_reward_action = self.reward_for_action_taken__action_limiting(config, section)
        else:
            raise ValueError('action_reward_type {} not recognized'.format(action_reward_type))

    def __call__(self, achieved_goal=None, desired_goal=None, info={}):
        action = info['action']
        reward_for_achieved_goal = self.compute_reward_achieved_goal(achieved_goal, desired_goal)
        reward_for_action_taken = self.compute_reward_action(action)
        return reward_for_achieved_goal + reward_for_action_taken

    class zero_fn(object):

        def __init__(self, config, section):
            pass

        def __call__(self, *args, **kwargs):
            return 0.0

    class reward_for_achieved_goal__linear(object):

        def __init__(self, config, section):
            self.alpha = config.getfloat(section, 'alpha')

        def __call__(self, achieved_goal, desired_goal):
            dist = goal_distance(achieved_goal, desired_goal)
            return -1.0 * self.alpha * dist

    class reward_for_achieved_goal__quadratic(object):

        def __init__(self, config, section):
            self.alpha = config.getfloat(section, 'alpha')

        def __call__(self, achieved_goal, desired_goal):
            dist = goal_distance(achieved_goal, desired_goal)
            return -1.0 * self.alpha * dist ** 2

    class reward_for_achieved_goal__smooth_abs(object):

        def __init__(self, config, section):
            self.alpha = config.getfloat(section, 'alpha')

        def __call__(self, achieved_goal, desired_goal):
            assert achieved_goal.shape == desired_goal.shape
            delta = achieved_goal - desired_goal
            alpha = self.alpha
            return -1.0 * (np.sqrt(np.mean(np.square(delta)) + alpha ** 2) - alpha)

    class reward_for_action_taken__quadratic(object):

        def __init__(self, config, section):
            self.beta = config.getfloat(section, 'beta')

        def __call__(self, action):
            return -1.0 * self.beta * np.linalg.norm(action)

    class reward_for_action_taken__action_limiting(object):

        def __init__(self, config, section):
            self.beta = config.getfloat(section, 'beta')

        def __call__(self, action):
            beta = self.beta
            return -1.0 * beta ** 2 * (np.mean(np.cosh(action / beta)) - 1)


class SparseGoalRewardFunc(RewardFunc):

    def __init__(self, config, section):
        super().__init__(config, section)
        self.goal_radius = config.getfloat(section, 'goal_radius')

    def __call__(self, achieved_goal, desired_goal, action=None):
        dist = goal_distance(achieved_goal, desired_goal)
        if dist < self.goal_radius:
            return 0.0
        else:
            return -1.0
