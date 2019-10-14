"""Inspect MuJoCo DDPG results. Ideally we can use similar stuff for gym-cloth.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import argparse
import csv
import os
import sys
import pickle
import numpy as np
import pandas as pd
from os.path import join
from collections import defaultdict

# matplotlib
titlesize = 33
xsize = 30
ysize = 30
ticksize = 25
legendsize = 25
er_alpha = 0.25

ENVS = [
    'Ant-v2',
    'HalfCheetah-v2',
    'Hopper-v2',
    'InvertedPendulum-v2',
    'Reacher-v2',
    'Swimmer-v2',
    'Walker2d-v2',
]


def smoothed(x, w):
    """Smooth x by averaging over sliding windows of w, assuming sufficient length.
    """
    if len(x) <= w:
        return x
    smooth = []
    for i in range(1, w):
        smooth.append( np.mean(x[0:i]) )
    for i in range(w, len(x)+1):
        smooth.append( np.mean(x[i-w:i]) )
    assert len(x) == len(smooth), "lengths: {}, {}".format(len(x), len(smooth))
    return np.array(smooth)


def _get_jagged_mean_std(data, w=-1):
    """Mean and std vectors for a 'jagged' set of data.

    Thus, just doing an `np.array()` on a list of lists won't usually give the
    desired result. I have to compute means/stdevs explicitly here.

    Parameters
    ----------
    data: list
        List of lists containing things of which we want to take means/stdevs.
    """
    mean_vec = []
    std_vec = []
    maxlen = -1
    for item in data:
        maxlen = max(maxlen, len(item))
    for idx in range(maxlen):
        vals = []
        for item in data:
            if idx < len(item):
                vals.append(item[idx])
        mean_vec.append(np.mean(vals))
        std_vec.append(np.std(vals))
    if w != -1:
        mean_vec = smoothed(mean_vec, w=w)
        std_vec = smoothed(std_vec, w=w)
    return np.array(mean_vec), np.array(std_vec)


def _get_stuff_from_monitor(mon):
    """Get stuff from `monitor` log files.

    Monitor files are named `0.envidx.monitor.csv` and have one line for each
    episode that finished in that CPU 'core', with the reward, length (number
    of steps) and the time (in seconds). The lengths are not cumulative, but
    time is cumulative.
    """
    scores = []
    steps  = []
    times  = []
    with open(mon, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for csv_row in csv_reader:
            # First two lines don't contain interesting stuff.
            if line_count == 0 or line_count == 1:
                line_count += 1
                continue
            scores.append(float(csv_row[0]))
            steps.append(int(csv_row[1]))
            times.append(float(csv_row[2]))
            line_count += 1
    print("finished: {}".format(mon))
    return scores, steps, times


def plot_mujoco(args, env_to_directory):
    """Plot from monitor files.
    """
    nrows, ncols = len(env_to_directory), 2
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharey='row',
                           figsize=(13*ncols,6*nrows))

    for idx,env in enumerate(sorted(env_to_directory.keys())):
        print('plotting: ', env)
        progress = []
        for directory in env_to_directory[env]:
            progfile = join('/tmp', directory, 'progress.csv')
            df = pd.read_csv(progfile, delimiter=',')
            expl_data = df['rollout/return_history'].tolist()
            eval_data = df['eval/return_history'].tolist()
            expl_l = '{}'.format(directory)
            eval_l = '{}'.format(directory)
            ax[idx,0].plot(expl_data, label=expl_l)
            ax[idx,1].plot(eval_data, label=eval_l)
        ax[idx,0].set_title('{}: Exploration'.format(env), fontsize=titlesize)
        ax[idx,1].set_title('{}: Evaluation'.format(env), fontsize=titlesize)

    for row in range(nrows):
        for col in range(ncols):
            # https://github.com/BerkeleyAutomation/baselines-fork/issues/5
            ax[row,col].set_xlabel("Evaluation Points", fontsize=ysize)
            ax[row,col].set_ylabel("Avg 100 Episode Return", fontsize=ysize)
            ax[row,col].tick_params(axis='x', labelsize=ticksize)
            ax[row,col].tick_params(axis='y', labelsize=ticksize)
            leg = ax[row,col].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
    figname = args.title
    plt.tight_layout()
    plt.savefig(figname)
    print("Just saved: {}\n".format(figname))


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument('--title', type=str, default='mujoco.png')
    args = pp.parse_args()

    # All OpenAI monitor files.
    openai_files = sorted(
        [x for x in os.listdir('/tmp') if 'openai-2019-06-' in x]
    )

    # Go through and get all the monitors for each game.
    env_to_directory  = defaultdict(list)
    for openai_f in openai_files:
        logdir = join('/tmp', openai_f, 'log.txt')
        with open(logdir) as fh:
            _ = fh.readline()
            line = fh.readline().strip()
            line_sp = line.split()
            mujoco_env = line_sp[-1]
            assert mujoco_env in ENVS, "{}, {}, {}".format(mujoco_env, line, openai_f)
            env_to_directory[mujoco_env].append(openai_f)

    print('env to directory:')
    for env in env_to_directory:
        print(env_to_directory[env], ': ', env)

    plot_mujoco(args, env_to_directory)
