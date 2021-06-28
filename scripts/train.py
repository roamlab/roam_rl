import argparse
from roam_rl.utils.path_utils import get_experiment_dir, get_config_path
from roam_rl.utils.config_utils import ConfigParser, initfromconfig
import os

def main(args):

    config_file = args.config_file
    config = ConfigParser()
    config.read(config_file)
    experiment_no = config.get('experiment', 'experiment_no')
    os.makedirs(os.environ['EXPERIMENTS_DIR'], exist_ok=True)
    experiment_dir = get_experiment_dir(os.environ['EXPERIMENTS_DIR'], experiment_no, mkdir=True)
    config_path = get_config_path(experiment_dir, experiment_no)
    config.save(config_path)
    algo = initfromconfig(config, config.get('experiment', 'algo'))
    algo.set_experiment_dir(experiment_dir)
    algo.train(model_path=args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='A string specifying the path to a config file')
    parser.add_argument('--model_path', help='path to the model to be loaded')
    arg = parser.parse_args()
    main(arg)

