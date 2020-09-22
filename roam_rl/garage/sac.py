import gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import deterministic, LocalRunner
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC as _SAC_
from garage.torch.policies import TanhGaussianMLPPolicy

from garage.torch.q_functions import ContinuousMLPQFunction
import os
from confac import make


class SAC:

    def __init__(self, config, section):

        # Setup
        self.experiment_dir = None
        self.config = config
        self.section = section
        self.env_maker = make(config, config.get(section, 'env_maker'))
        self.seed = config.getint(section, 'seed')
        self.snapshot_mode = config.get(section, 'snapshot_mode', fallback='last')
        
        # SAC hyper parameters 
        self.policy_hidden_sizes = eval(config.get(section, 'policy_hidden_sizes', fallback='[256, 256]'))
        self.qf_hidden_sizes = eval(config.get(section, 'qf_hidden_sizes', fallback = '[256, 256]'))
        self.buffer_capacity_in_transitions = int(config.getfloat(section, 'buffer_capacity_in_transitions', fallback=1e6))
        self.gradient_steps_per_itr = config.getint(section, 'gradient_steps_per_iteration', fallback=1000)
        self.max_path_length = config.getint(section, 'max_path_length', fallback=1000)
        self.max_eval_path_length = config.getint(section, 'max_eval_path_length', fallback=1000)
        self.min_buffer_size = int(config.getfloat(section, 'min_buffer_size', fallback=1e4))
        self.target_update_tau = config.getfloat(section, 'target_update_tau', fallback=5e-3)
        self.discount = config.getfloat(section, 'discount',  fallback=0.99)
        self.buffer_batch_size = config.getint(section, 'buffer_batch_size', fallback=256)
        self.reward_scale = config.getfloat(section, 'reward_scale', fallback=1.)
        self.steps_per_epoch = config.getint(section, 'steps_per_epoch', fallback=1)

    def set_experiment_dir(self, experiment_dir):
        self.experiment_dir = experiment_dir

    def train(self):
        
        # define
        @wrap_experiment(snapshot_mode=self.snapshot_mode, log_dir=self.experiment_dir)
        def sac(ctxt=None):
            """ Set up environment and algorithm and run the task.

            Args:
                ctxt (garage.experiment.ExperimentContext): The experiment
                    configuration used by LocalRunner to create the snapshotter.
                seed (int): Used to seed the random number generator to produce
                    determinism.

            """
            deterministic.set_seed(self.seed)
            runner = LocalRunner(snapshot_config=ctxt)
            env = GarageEnv(normalize(self.env_maker()))

            policy = TanhGaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=self.policy_hidden_sizes,
                hidden_nonlinearity=nn.ReLU,
                output_nonlinearity=None,
                min_std=np.exp(-20.),
                max_std=np.exp(2.),
            )

            qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                        hidden_sizes=self.qf_hidden_sizes,
                                        hidden_nonlinearity=F.relu)

            qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                        hidden_sizes=self.qf_hidden_sizes,
                                        hidden_nonlinearity=F.relu)

            replay_buffer = PathBuffer(capacity_in_transitions=self.buffer_capacity_in_transitions)

            algo = _SAC_(env_spec=env.spec,
                    policy=policy,
                    qf1=qf1,
                    qf2=qf2,
                    gradient_steps_per_itr=self.gradient_steps_per_itr,
                    max_path_length=self.max_path_length,
                    max_eval_path_length=self.max_eval_path_length,
                    replay_buffer=replay_buffer,
                    min_buffer_size=self.min_buffer_size,
                    target_update_tau=self.target_update_tau,
                    discount=self.discount,
                    buffer_batch_size=self.buffer_batch_size,
                    reward_scale=self.reward_scale,
                    steps_per_epoch=self.steps_per_epoch)

            if torch.cuda.is_available():
                set_gpu_mode(True)
            else:
                set_gpu_mode(False)
            algo.to()
            
            runner.setup(algo=algo, env=env, sampler_cls=LocalSampler)
            runner.train(n_epochs=10, batch_size=1000)
        
        # call
        sac()


