"""Get results for rolling out trained policies, including the BC portion.

It's going to require some manual work, unfortunately, to basically make the
directories we need. But put them on nfs/diskstation and it will probably be
easier. We probably should use a backup somewhere, though! Look at:

$ ls -lh /nfs/diskstation/seita/clothsim/
drwxr-xr-x 1 nobody nogroup 1.9K Aug 13 14:08 rollouts_4.0.1
drwxr-xr-x 1 nobody nogroup 1.9K Aug 13 14:15 rollouts_4.0.2

and similar files. Example:

    python scripts/plot_rollouts.py /nfs/diskstation/seita/clothsim/rollouts_6.1.0c/

Keep this for one set of results. For combining, see `combo_rollouts.py`.

Look at `baselines/run.py` for details on how we stored information.
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
titlesize = 42
xsize = 40
ysize = 40
ticksize = 36
legendsize = 36
er_alpha = 0.25
bar_width = 0.3
bar_alpha = 0.9
lw = 5
EPS = 0.02


def _get_tier(files):
    """We can do it programmatically.

    files is a list of pickle files.
    """
    first_pickle_file = files[0]
    if 'tier1' in first_pickle_file:
        for f in files:
            assert 'tier1' in f, f
        return 'Tier 1'
    elif 'tier2' in first_pickle_file:
        for f in files:
            assert 'tier2' in f, f
        return 'Tier 2'
    elif 'tier3' in first_pickle_file:
        for f in files:
            assert 'tier3' in f, f
        return 'Tier 3'
    else:
        raise ValueError(first_pickle_file)


def _demo_coverage(tier):
    if tier == 'Tier 1':
        return 0.96
    elif tier == 'Tier 2':
        return 0.95
    elif tier == 'Tier 3':
        return 0.95
    else:
        raise ValueError(tier)


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


def plot(args, rollout_data, files, success_thresh = 0.92):
    """Plot rollout data."""
    stats = defaultdict(list)
    stats_bc = defaultdict(list)
    stats_d = defaultdict(list)
    tier = _get_tier(files)
    avg_demo = _demo_coverage(tier)

    # I saved these keys: 'load_path', 'r', 'c', 'iv', 'oob', 's_c', 's_iv'
    for r_tuple in rollout_data:
        # Each 'r_tuple' consists of one checkpoint's rolled out data.
        r_chkpt, r_data, r_file = r_tuple
        stats['chkpt'].append(r_chkpt)
        if 'bc_' in r_chkpt:
            stats_bc['chkpt'].append(r_chkpt)
        elif 'dagger_' in r_chkpt:
            stats_d['chkpt'].append(r_chkpt)
        else:
            raise ValueError(r_chkpt)

        # Each of the `key`s in `r_data[key]` is a full list, because it came
        # from the 'r_tuple' which is from a full set of 100-ish rollouts. Take
        # mean/std to get a single number for this checkpoint's performance.
        c = np.array(r_data['c']) # We can use for success thresholding.

        for key in r_data.keys():
            if key == 'load_path':
                continue

            # Eh I prefer splitting into bc vs dagger? For now.
            stats[key+'_mean'].append( np.mean(r_data[key]) )
            stats[key+'_std'].append( np.std(r_data[key]) )
            if key == 'c':
                success = np.sum(c >= success_thresh) / len(c)
                stats['success'].append( success )

            if 'bc_' in r_chkpt:
                stats_bc[key+'_mean'].append( np.mean(r_data[key]) )
                stats_bc[key+'_std'].append( np.std(r_data[key]) )
                if key == 'c':
                    success = np.sum(c >= success_thresh) / len(c)
                    stats_bc['success'].append( success )

            elif 'dagger_' in r_chkpt:
                stats_d[key+'_mean'].append( np.mean(r_data[key]) )
                stats_d[key+'_std'].append( np.std(r_data[key]) )
                if key == 'c':
                    success = np.sum(c >= success_thresh) / len(c)
                    stats_d['success'].append( success )

            else:
                raise ValueError(r_chkpt)

    # Pretty sure this will be ok
    for key in stats_bc.keys():
        stats_bc[key] = np.array(stats_bc[key])
    for key in stats_d.keys():
        stats_d[key] = np.array(stats_d[key])

    # Inspect
    print('\nBehavior Cloning')
    print('chkpt, mean(start), mean(end), success%')
    for (chk, s_c, c, success) in zip(stats_bc['chkpt'], stats_bc['s_c_mean'],
                                      stats_bc['c_mean'], stats_bc['success']):
        print('  {}, {:.2f}, {:.2f}, {:.2f}'.format(chk, s_c, c, success))
    print('\nDAgger')
    print('chkpt, mean(start), mean(end), success%')
    for (chk, s_c, c, success) in zip(stats_d['chkpt'], stats_d['s_c_mean'],
                                      stats_d['c_mean'], stats_d['success']):
        print('  {}, {:.2f}, {:.2f}, {:.2f}'.format(chk, s_c, c, success))

    # Plot !! NOTE: might adjust the widths here.
    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharey='row',
                           figsize=(10*ncols, 8*nrows),
                           gridspec_kw={'width_ratios': [1, 1.5]})

    # Coverage.
    #ax[0,0].set_title('Coverage'.format(), fontsize=titlesize)
    #ax[0,0].set_ylabel('Coverage'.format(), fontsize=ysize)
    #ax[0,0].set_ylim([0-EPS,1+EPS])
    #ydata_start = [x for x in stats['s_c_mean']]
    #ydata_end   = [x for x in stats['c_mean']]
    #assert len(ydata_start) == len(ydata_end), len(ydata_start)
    #x = np.arange( len(ydata_start) )

    ax[0,0].set_title('Test Rollouts, BC, {}'.format(tier), fontsize=titlesize)
    ax[0,0].set_ylabel('Final Coverage'.format(), fontsize=ysize)
    ax[0,0].set_xlabel('BC Training Epochs'.format(), fontsize=xsize)
    ax[0,0].set_ylim([0-EPS,1+EPS])

    ax[0,1].set_title('Test Rollouts, DAgger, {}'.format(tier), fontsize=titlesize)
    #ax[0,1].set_ylabel('Final Coverage'.format(), fontsize=ysize)
    ax[0,1].set_xlabel('DAgger Training Iterations'.format(), fontsize=xsize)
    ax[0,1].set_ylim([0-EPS,1+EPS])

    # Assume it's the last coordinate.
    def _get_xcoords(checkpoints):
        x = []
        for ch in checkpoints:
            split = ch.split('_')
            x.append( int(split[-1]) )
        return x

    x_bc = _get_xcoords(stats_bc['chkpt'])
    x_d = _get_xcoords(stats_d['chkpt'])
    ax[0,0].plot(x_bc, stats_bc['c_mean'], lw=lw, marker='x', ms=18, mew=5)
                 #label='Final Coverage')
    maxval = max( np.max(stats_bc['c_mean']), np.max(stats_d['c_mean']) )
    ax[0,1].plot(x_d, stats_d['c_mean'], lw=lw, marker='x', ms=18, mew=5,
                 label='Final Coverage (Best: {:.3f})'.format(maxval))

    ax[0,0].fill_between(x_bc,
                         stats_bc['c_mean'] - stats_bc['c_std'],
                         stats_bc['c_mean'] + stats_bc['c_std'],
                         alpha=er_alpha)
    ax[0,1].fill_between(x_d,
                         stats_d['c_mean'] - stats_d['c_std'],
                         stats_d['c_mean'] + stats_d['c_std'],
                         alpha=er_alpha)

    avg_start = np.mean(stats_bc['s_c_mean'])
    ax[0,0].axhline(y=avg_start, ls='--', lw=3, color='r')
                    #label='Start Coverage {:.3f}'.format(avg_start))
    ax[0,1].axhline(y=avg_start, ls='--', lw=3, color='r',
                    label='Start Coverage: {:.3f}'.format(avg_start))
    ax[0,0].axhline(y=avg_demo, ls='--', lw=3, color='b')
                    #label='Demo. Coverage {:.3f}'.format(avg_demo))
    ax[0,1].axhline(y=avg_demo, ls='--', lw=3, color='b',
                    label='Demo. Coverage: {:.3f}'.format(avg_demo))

    # Bells and whistles
    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
            # I think it's better to share axes in the x direction to be
            # consistent with steps, but doing so removes the axis ticks. This
            # reverts it so we get the ticks on all the axis.
            #ax[r,c].xaxis.set_tick_params(which='both', labelbottom=True)

    plt.tight_layout()
    figname = 'fig_rollout_{}.png'.format(args.base_path)
    plt.savefig(figname)
    print("Just saved: {}\n".format(figname))


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument('path', type=str)
    args = pp.parse_args()
    assert args.path is not None
    if args.path[-1] == '/':
        args.path = args.path[:-1]
    args.base_path = os.path.basename(args.path)

    files = [join(args.path, x) for x in os.listdir(args.path) if x[-4:] == '.pkl']
    rollout_data = []
    for f in files:
        with open(f, 'rb') as fh:
            data = pickle.load(fh)
            # Just to make it easier to sort by load path.  Also, apply desired
            # ordering for files to make life easier. Check that we loaded correctly.
            p_head, p_tail = os.path.split(data['load_path'])
            if 'bc_' not in p_tail:
                p_tail = 'dagger_{}'.format(p_tail)
            rollout_data.append( (p_tail, data, f) )

    rollout_data = sorted(rollout_data, key = lambda x : x[0])
    print('This is the order in which we plot:')
    for rd in rollout_data:
        pth, data, f = rd
        print(' loaded: {}  from: {}'.format(pth, data['load_path']))
    plot(args, rollout_data, files)
