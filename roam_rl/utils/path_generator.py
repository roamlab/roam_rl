import os
from roam_utils.provenance.directory_helpers import make_dir
from roam_utils.provenance import path_generator as pg


class PathGenerator(pg.PathGenerator):

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_ppo_experiment_dir(path_to_experiments, robot_name, experiment_no):
        experiment_dir = os.path.join(path_to_experiments, 'experiments', robot_name, 'ppo',
                                      'experiment_' + str(experiment_no).zfill(2))
        make_dir(experiment_dir)
        return experiment_dir

    @staticmethod
    def get_ppo_her_experiment_dir(path_to_experiments, robot_name, experiment_no):
        experiment_dir = os.path.join(path_to_experiments, 'experiments', robot_name, 'ppo_her',
                                      'experiment_' + str(experiment_no).zfill(2))
        make_dir(experiment_dir)
        return experiment_dir

    @staticmethod
    def get_ppo_model_path(experiments_dir, seed):
        model_dir = os.path.join(experiments_dir, 'seed-{}'.format(seed))
        make_dir(model_dir)
        model_path = os.path.join(model_dir, 'model.pkl'.format(seed))
        return model_path

    @staticmethod
    def get_env_path(experiments_dir, seed):
        model_dir = os.path.join(experiments_dir, 'env')
        make_dir(model_dir)
        model_path = os.path.join(model_dir, 'env_seed_{}.pkl'.format(seed))
        return model_path

    @staticmethod
    def get_ppo_log_dir(experiments_dir, seed):
        logdir = os.path.join(experiments_dir, 'seed-{}'.format(seed))
        make_dir(logdir)
        return logdir

    @staticmethod
    def get_ppo_trajectory_dir(experiments_dir, trajectory_no):
        trajectory_dir = os.path.join(experiments_dir, 'trajectories', 'trajectory_dir_{}'.format(str(
            trajectory_no).zfill(5)))
        make_dir(trajectory_dir)
        return trajectory_dir

    @staticmethod
    def get_record_sim_dir(experiment_dir):
        record_sim_dir = os.path.join(experiment_dir, 'pngs')
        make_dir(record_sim_dir)
        return record_sim_dir

    @staticmethod
    def get_gui_render_path(save_dir, count):
        return os.path.join(save_dir, 'chain_gui_{}.png'.format(str(count).zfill(5)))

    @staticmethod
    def get_config_pathname(load_dir, experiment_no):
        return os.path.join(load_dir, 'config_' + str(experiment_no).zfill(2) + '.cfg')