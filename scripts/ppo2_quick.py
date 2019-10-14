"""Quickly inspect results, e.g., from the `/tmp/openai-*` directories.
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


def plot_monitor(args):
    """Plot from monitor files.
    """
    nrows, ncols = 2, 3
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharey='row',
                           figsize=(11*ncols,8*nrows))
    title = args.title

    # Global statistics across all monitors
    scores_all = []
    steps_all = []
    times_all = []
    total_tr_steps = 0
    train_hours = 0

    monitors = sorted(
        [x for x in os.listdir(args.path) if 'monitor.csv' in x and '.swp' not in x]
    )

    # First row, info from all the monitors, i.e., number of CPUs.
    for env_idx,mon in enumerate(monitors):
        monitor_path = join(args.path, mon)
        scores, steps, times = _get_stuff_from_monitor(monitor_path)

        # Now process to see as a function of episodes and training steps, etc.
        num_episodes = len(scores)
        tr_episodes = np.arange(num_episodes)
        tr_steps = np.cumsum(steps)
        tr_times = np.array(times) / 60.0 # get it in minutes

        # Plot for individual monitors.
        envlabel = 'env {}'.format(env_idx)
        sm_10 = smoothed(scores, w=10)
        ax[0,0].plot(tr_steps, sm_10, label=envlabel+'; {:.1f}'.format(sm_10[-1]))
        sm_100 = smoothed(scores, w=100)
        ax[0,1].plot(tr_times, sm_100, label=envlabel+'; {:.1f}'.format(sm_100[-1]))

        # Episode lengths
        sm_10 = smoothed(steps, w=10)
        ax[1,0].plot(tr_steps, sm_10, label=envlabel+'; {:.1f}'.format(sm_10[-1]))
        sm_100 = smoothed(steps, w=100)
        ax[1,1].plot(tr_times, sm_100, label=envlabel+'; {:.1f}'.format(sm_100[-1]))

        # Handle global stuff.
        total_tr_steps += tr_steps[-1]
        train_hours = max(train_hours, tr_times[-1] / 60.0)
        scores_all.append(scores)
        steps_all.append(steps)

    # More global stuff, but we can smooth this as well.
    scores_mean, scores_std = _get_jagged_mean_std(scores_all, w=5)
    steps_mean,  steps_std  = _get_jagged_mean_std(steps_all, w=5)
    xs = np.arange(len(scores_mean))
    k = 10
    label = 'Mean Ep. Rew., avg last {}: {:.1f}'.format(k, np.mean(scores_mean[-k:]))
    ax[0,2].plot(xs, scores_mean, label=label)
    ax[0,2].fill_between(xs, scores_mean+scores_std, scores_mean-scores_std,
                         color='blue', alpha=er_alpha)
    xs = np.arange(len(steps_mean))
    label = 'Mean Ep. Lengths, avg last {}: {:.1f}'.format(k, np.mean(steps_mean[-k:]))
    ax[1,2].plot(xs, steps_mean, label=label)
    ax[1,2].fill_between(xs, steps_mean+steps_std, steps_mean-steps_std,
                         color='red', alpha=er_alpha)

    # Bells and whistles
    for row in range(nrows):
        for col in range(ncols):
            if row == 0:
                ax[row,col].set_ylabel("Episode Reward", fontsize=30)
            elif row == 1:
                ax[row,col].set_ylabel("Episode Length", fontsize=30)
            if col == 0:
                ax[row,col].set_xlabel("Train Steps (total {})".format(total_tr_steps),
                        fontsize=xsize)
            elif col == 1:
                ax[row,col].set_xlabel("Train Time (Hours {:.2f})".format(train_hours),
                        fontsize=xsize)
            elif col == 2:
                ax[row,col].set_xlabel("Number of Episodes per Env".format(),
                        fontsize=xsize)
            ax[row,col].tick_params(axis='x', labelsize=ticksize)
            ax[row,col].tick_params(axis='y', labelsize=ticksize)
            leg = ax[row,col].legend(loc="best", ncol=2, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
    ax[0,0].set_title(title+', Smoothed (w=10)', fontsize=titlesize)
    ax[0,1].set_title(title+', Smoothed (w=100)', fontsize=titlesize)
    ax[0,2].set_title(title+', Rewards Averaged', fontsize=titlesize)
    ax[1,0].set_title(title+', Smoothed (w=10)', fontsize=titlesize)
    ax[1,1].set_title(title+', Smoothed (w=100)', fontsize=titlesize)
    ax[1,2].set_title(title+', Lengths Averaged', fontsize=titlesize)
    plt.tight_layout()
    figname = '{}_monitor.png'.format(title)
    plt.savefig(figname)
    print("Just saved: {}\n".format(figname))


def plot_progress(args):
    """Plot from the progress csv file.

    Ignore some of these to save space. For example, serial_timesteps can be
    exactly recovered from total_timesteps by dividing that by the number of
    environments.
    """
    col_to_skip = ['fps', 'nupdates', 'serial_timesteps', 'time_elapsed']
    progfile = join(args.path, 'progress.csv')
    df = pd.read_csv(progfile, delimiter = ',')
    print("loaded csv, with shape {}".format(df.shape))
    print("skipping these columns: {}".format(col_to_skip))

    # Next, get the plot set up, one row per statistic?
    nrows, ncols = df.shape[1]-len(col_to_skip), 1
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharey='row',
                           figsize=(13*ncols,4*nrows))
    title = args.title

    k= 10
    row = 0
    for column in df:
        if column in col_to_skip:
            continue
        data = df[column].tolist()
        label = 'avg last {}, {:.3f}'.format(k, np.mean(data[-k:]))
        ax[row,0].plot(data, label=label)
        ax[row,0].set_ylabel(column, fontsize=ysize)
        ax[row,0].tick_params(axis='x', labelsize=ticksize)
        ax[row,0].tick_params(axis='y', labelsize=ticksize)
        leg = ax[row,0].legend(loc="best", ncol=1, prop={'size':legendsize})
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)
        row += 1

    plt.tight_layout()
    figname = '{}_progress.png'.format(title)
    plt.savefig(figname)
    print("Just saved: {}\n".format(figname))


def plot_actions(args):
    """Plot from the `actions` directory.

    For each batch of data collection (rollouts through all parallel envs) we
    get the actions that were taken in that batch (and only that batch). Then
    we can collect statistics about the distribution fo actions at this batch.
    Repeat for batches at all time steps.
    """
    act_dir = join(args.path, 'actions')
    act_files = sorted([x for x in os.listdir(act_dir) if '.pkl' in x])
    stats = defaultdict(list)

    for a_file in act_files:
        with open(join(act_dir,a_file), 'rb') as fh:
            data = pickle.load(fh)
            assert len(data.shape) == 2, data.shape
            assert data.shape[1] == 4, data.shape
            # There's one element per action in axis=1, thus we want to
            # compute statistics *over* axis=0, ---> shape (4,) result.
            stats['mean'].append( np.mean(data, axis=0) )
            stats['std'].append( np.std(data, axis=0) )
            stats['10'].append( np.percentile(data, q=10, axis=0) )
            stats['90'].append( np.percentile(data, q=90, axis=0) )
    print("Loaded {} action data files.".format(len(act_files)))
    for key in stats:
        stats[key] = np.array(stats[key])
        print(stats[key].shape, key)

    nrows, ncols = 4, 1
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharey='row',
                           figsize=(13*ncols,4*nrows))
    title = args.title

    # TODO need to programmatically determine if length/angle or deltas.
    action_names = ['x', 'y', 'length', 'angle']
    colors = ['blue', 'red', 'black', 'gold']
    xs = np.arange(stats['mean'].shape[0])

    for rr in range(nrows):
        a_mean = stats['mean'][:,rr]
        labels = [
            'mean action (last {:.2f})'.format(a_mean[-1]),
            '10th perc. (last {:.2f})'.format(stats['10'][:,rr][-1]),
            '90th perc. (last {:.2f})'.format(stats['90'][:,rr][-1]),
        ]
        ax[rr,0].plot(xs, a_mean, color=colors[0], label=labels[0])
        ax[rr,0].plot(xs, stats['10'][:,rr], color=colors[1], label=labels[1])
        ax[rr,0].plot(xs, stats['90'][:,rr], color=colors[2], label=labels[2])
        ax[rr,0].fill_between(xs,
                a_mean + stats['std'][:,rr],
                a_mean - stats['std'][:,rr],
                color=colors[3],
                alpha=er_alpha)
        ytext = 'Action: {}'.format(action_names[rr])
        ax[rr,0].set_ylabel(ytext, fontsize=ysize)
        leg = ax[rr,0].legend(loc="best", ncol=1, prop={'size':legendsize})
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)
        ax[rr,0].tick_params(axis='x', labelsize=ticksize)
        ax[rr,0].tick_params(axis='y', labelsize=ticksize)
        ax[rr,0].set_xlabel('Average per nupdate (i.e., batch of data)', fontsize=xsize)

    plt.tight_layout()
    figname = '{}_actions.png'.format(title)
    plt.savefig(figname)
    print("Just saved: {}\n".format(figname))


def plot_ep_infos(args):
    """Plot from the `ep_all_infos` directory.
    """
    ep_dir = join(args.path, 'ep_all_infos')
    ep_files = sorted([x for x in os.listdir(ep_dir) if '.pkl' in x])
    print("Loaded {} episode pickle files.".format(len(ep_files)))
    stats = defaultdict(list)

    for e_file in ep_files:
        with open(join(ep_dir,e_file), 'rb') as fh:
            # A LIST of info stats from COMPLETED episodes.
            data = pickle.load(fh)
            assert len(data) > 0 # for now
            coverage = []
            oob = []
            tears = []
            steps = []

            # For each episode, get stats.
            for epis in data:
                # Ending coverage, except if tear/oob (then it's the one before that).
                coverage.append(epis['actual_coverage'])
                oob.append(int(epis['out_of_bounds']))
                tears.append(int(epis['have_tear']))
                steps.append(epis['num_steps'])

            # Now accumulate them in batches, one data point per `nupdate`.
            stats['coverage_mean'].append( np.mean(coverage) )
            stats['coverage_std'].append( np.std(coverage) )
            stats['out_of_bounds'].append( np.mean(oob) )
            stats['have_tear'].append( np.mean(tears) )
            stats['num_steps'].append( np.mean(steps) )

    for key in stats:
        stats[key] = np.array(stats[key])
        print(stats[key].shape, key)

    # Plot !!
    nrows, ncols = 4, 1
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharey='row',
                           figsize=(13*ncols,4*nrows))
    title = args.title
    stat_names = ['coverage_mean', 'out_of_bounds', 'have_tear', 'num_steps']
    colors = ['blue', 'red', 'black', 'gold']
    xs = np.arange(stats['coverage_mean'].shape[0])

    for (rr, key) in enumerate(stat_names):
        label = '{}, last {:.2f}'.format(stat_names[rr], stats[key][-1])
        ax[rr,0].plot(xs, stats[key], color=colors[rr], label=label)
        if key == 'coverage_mean':
            ax[rr,0].fill_between(xs,
                    stats['coverage_mean'] + stats['coverage_std'],
                    stats['coverage_mean'] - stats['coverage_std'],
                    color=colors[rr],
                    alpha=er_alpha)
        # Bells and whistles.
        if key != 'num_steps':
            eps = 0.1
            ax[rr,0].set_ylim([0-eps,1+eps])
        ytext = '{}'.format(stat_names[rr])
        ax[rr,0].set_ylabel(ytext, fontsize=ysize)
        leg = ax[rr,0].legend(loc="best", ncol=1, prop={'size':legendsize})
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)
        ax[rr,0].tick_params(axis='x', labelsize=ticksize)
        ax[rr,0].tick_params(axis='y', labelsize=ticksize)
        ax[rr,0].set_xlabel('Average per nupdate (i.e., batch of data)', fontsize=xsize)

    plt.tight_layout()
    figname = '{}_ep_infos.png'.format(title)
    plt.savefig(figname)
    print("Just saved: {}\n".format(figname))


def plot_logstds(args):
    """Plot from the `logstd` directory.
    """
    std_dir = join(args.path, 'logstd')
    std_files = sorted([x for x in os.listdir(std_dir) if '.pkl' in x])
    print("Loaded {} log std pickle files.".format(len(std_files)))
    stats = defaultdict(list)
    logstds = []

    for s_file in std_files:
        # Each `s_file` is from a particular `nupdate` in the RL code.
        with open(join(std_dir,s_file), 'rb') as fh:
            data = pickle.load(fh)
        assert len(data) > 0 # for now
        logstds.append(data)
    logstds = np.squeeze(np.array(logstds))
    raw_stds = np.exp(logstds)
    print("Loaded log stdev's, shape: {}".format(logstds.shape))

    # Set up figure.
    nrows, ncols = 2, 1
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=(14*ncols,4*nrows))
    title = args.title
    action_names = ['x', 'y', 'length', 'angle']
    colors = ['blue', 'red', 'black', 'gold']
    xs = np.arange(logstds.shape[0])

    # Plot log std's and exp(log std's) = std's.
    for a_idx,act in enumerate(action_names):
        label0 = '(last {:.2f}) {}'.format((logstds[:,a_idx])[-1], act)
        ax[0,0].plot(xs, logstds[:,a_idx], lw=2, label=label0, color=colors[a_idx])
        label1 = '(last {:.2f}) {}'.format((raw_stds[:,a_idx])[-1], act)
        ax[1,0].plot(xs, raw_stds[:,a_idx], lw=2, label=label1, color=colors[a_idx])

    # Bells and whistles.
    ax[0,0].set_ylabel('Log Std', fontsize=xsize)
    ax[1,0].set_ylabel('Raw Std', fontsize=xsize)
    for rr in range(nrows):
        leg = ax[rr,0].legend(loc="best", ncol=1, prop={'size':legendsize})
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)
        ax[rr,0].tick_params(axis='x', labelsize=ticksize)
        ax[rr,0].tick_params(axis='y', labelsize=ticksize)
        ax[rr,0].set_xlabel('Average per nupdate (i.e., batch of data)', fontsize=xsize)
    plt.tight_layout()
    figname = '{}_log_stds.png'.format(title)
    plt.savefig(figname)
    print("Just saved: {}\n".format(figname))


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument('--path', type=str)
    pp.add_argument('--title', type=str)
    args = pp.parse_args()
    assert args.path is not None
    assert args.title is not None
    # Each of these makes its own figure.
    plot_monitor(args)
    plot_progress(args)
    plot_actions(args)
    plot_ep_infos(args)
    plot_logstds(args)
