import os

def _mkdir(path):
    if not os.path.exists(path): os.mkdir(path)

def get_experiment_dir(path_to_experiments, experiment_no, mkdir=False):
    experiment_dir = os.path.join(path_to_experiments, 'experiment_'+ str(experiment_no).zfill(2))
    if mkdir: _mkdir(experiment_dir)
    return experiment_dir

def get_model_path(experiments_dir, seed, model_checkpoint=None):
    model_dir = os.path.join(experiments_dir, 'seed-{}'.format(seed))
    if model_checkpoint:
        model_path = os.path.join(model_dir, 'checkpoints/{}'.format(str(model_checkpoint).zfill(5)))
    else:
        model_path = os.path.join(model_dir, 'model_{}.pkl'.format(seed))
    return model_path

def get_log_dir(experiments_dir, seed):
    logdir = os.path.join(experiments_dir, 'seed-{}'.format(seed))
    return logdir

def get_config_path(load_dir, experiment_no):
    return os.path.join(load_dir, 'config_' + str(experiment_no).zfill(2) + '.cfg')