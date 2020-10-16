import os, sys
from confac import ConfigParser
from roam_rl import utils
from roam_rl.baselines.ppo import PPO
import matplotlib.pyplot as plt
import argparse
import numpy as np
from baselines.common import plot_util


def main(args):
    config_file = args.config_file
    path_to_experiments = os.environ["EXPERIMENTS_DIR"]

    run_config = ConfigParser()
    run_config.read(config_file)
    experiment_no = run_config.get('experiment', 'experiment_no')

    experiment_dir = utils.get_experiment_dir(path_to_experiments, experiment_no)
    config_file = utils.get_config_path(experiment_dir, experiment_no)
    config = ConfigParser()
    config.read(config_file)

    plot_training(experiment_no, experiment_dir)


def plot_training(experiment_no, experiment_dir):
    results = plot_util.load_results(experiment_dir, verbose=True)
    results.sort(key=lambda x: x.dirname)
    fig, ax = plt.subplots()
    color_set = ['b', 'g', 'r', 'm', 'y', 'c']
    ax.grid(color='#dddddd', linestyle='-', linewidth=1)
    for i, r in enumerate(results):
        i_ = i % len(color_set)
        plt.plot(np.cumsum(r.monitor.l), plot_util.smooth(r.monitor.r, radius=200), color=color_set[i_])
    for i, r in enumerate(results):
        i_ = i % len(color_set)
        plt.plot(np.cumsum(r.monitor.l), r.monitor.r, alpha=0.1, color=color_set[i_])
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.set_xlim(left=0)
    plt.tight_layout(pad=2)
    plt.xlabel('time steps')
    plt.ylabel('episode_reward')
    plt.title('experiment_{}'.format(experiment_no), fontsize=10)
    plt.legend([os.path.basename(r.dirname) for r in results])
    plt.savefig(os.path.join(experiment_dir, 'training_{}.png'.format(experiment_no)))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='A string specifying the path to a config file')
    main(parser.parse_args())


