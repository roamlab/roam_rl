import argparse
from roam_rl import utils
from roam_rl.baselines.ppo import PPO
import os
from confac import ConfigParser


def main(args):
    run_config_file = args.config_file
    model_checkpoint = args.model_checkpoint
    run_config = ConfigParser()
    run_config.read(run_config_file)
    experiment_no = run_config.get('experiment', 'experiment_no')
    experiment_dir = utils.get_experiment_dir(os.environ["EXPERIMENTS_DIR"], experiment_no)
    load_model_seed = run_config.get('experiment', 'load_model_seed')
    stochastic = run_config.getboolean('experiment', 'stochastic', fallback=False)

    env_seed = run_config.getint('experiment', 'env_seed')
    copy_sections = run_config.getlist('experiment', 'copy_sections')

    config_file = utils.get_config_path(experiment_dir, experiment_no)
    assert os.path.exists(config_file), 'config file does not exist in experiment dir'

    config = ConfigParser()
    config.read(config_file)
    for section in copy_sections:
        run_config.dump_section(section, recursive=True, dump=config)

    algo = PPO(config, config.get('experiment', 'algo'))
    algo.set_experiment_dir(experiment_dir)

    model, env = algo.load(model_seed=load_model_seed, model_checkpoint=model_checkpoint, env_seed=env_seed)
    algo.run(model=model, env=env, stochastic=stochastic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='A string specifying the path to a config file')
    parser.add_argument('-c', dest='model_checkpoint', type=int, help='Optional: From which checkpoint to load the model from')
    arg = parser.parse_args()
    main(arg)
