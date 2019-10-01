import time
import configparser
import argparse
from roam_rl.utils.path_generator import PathGenerator
import roam_utils.provenance.provenance as provenance
from roam_utils.logging import roam_logger
import logging
import socket
from datetime import datetime
from roam_rl.openai_baselines.ppo_her import PPOHER
import os


def main(args):
    config_file = args.config_file
    config_data = configparser.ConfigParser()
    config_data.read(config_file)
    experiment_no = config_data.get('experiment', 'experiment_no')
    robot_name = config_data.get('experiment', 'robot_name')
    experiment_dir = PathGenerator.get_ppo_her_experiment_dir(os.environ['EXPERIMENTS_DIR'], robot_name, experiment_no)
    provenance.save_svn(experiment_dir, experiment_no)
    provenance.save_config(config_data, experiment_dir, experiment_no)

    # setup roam logging
    base_level = config_data.getint('logging', 'script_base_level')
    detail_level = config_data.getint('logging', 'script_detail_level')
    console_level = config_data.getint('logging', 'script_console_level')
    debug_level = config_data.getint('logging', 'script_debug_level')
    roam_logger.setup_roam_logger(experiment_dir, experiment_no, base_level, detail_level, debug_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    logging.getLogger().addHandler(console_handler)
    logger = roam_logger.get_roam_logger()

    logger.log(base_level, 'machine: {}'.format(socket.gethostname()))
    start_time = time.time()
    logger.log(base_level, "Begin training, time: {}".format(datetime.now().strftime("%x %X")))

    ppo_section_name = config_data.get('experiment', 'ppo')
    ppo = PPOHER(config_data, ppo_section_name)
    ppo.set_experiment_dir(experiment_dir)
    ppo.learn()

    logger.log(base_level, "Training completed, time: {}".format(datetime.now().strftime("%x %X")))
    logger.log(base_level, 'Duration, time : {}s'.format(round(time.time() - start_time, 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='A string specifying the path to a config file')
    arg = parser.parse_args()
    main(arg)

