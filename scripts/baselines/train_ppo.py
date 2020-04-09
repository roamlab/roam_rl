import argparse
from roam_rl.baselines.ppo import PPO
from confac import ConfigParser
import os

def main(args):

    config_file = args.config_file

    config = ConfigParser()
    config.read(config_file)
    experiment_no = config.get('experiment', 'experiment_no')
    robot_name = config.get('experiment', 'robot_name')
    experiment_dir = PPO.get_experiment_dir(os.environ['EXPERIMENTS_DIR'], robot_name, experiment_no)
    config_path = PPO.get_config_path(experiment_dir, experiment_no)
    config.save(config_path)

    ppo_section = config.get('experiment', 'ppo')
    ppo = PPO(config, ppo_section)
    ppo.set_experiment_dir(experiment_dir)
    ppo.learn()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='A string specifying the path to a config file')
    arg = parser.parse_args()
    main(arg)

