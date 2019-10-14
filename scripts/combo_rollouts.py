"""Use this for combining rollouts together.

For individual runs, see `scripts/plot_rollouts.py`.

For the 'DAgger improvement over BC phase', I just used this:

  In [9]: e1
  Out[9]: [88.45, 94.84]

  In [10]: e2
  Out[10]: [89.43, 89.56]

  In [11]: e3
  Out[11]: [84.26, 91.24]

  In [12]: e4
  Out[12]: [76.69, 84.03]

  In [13]: e5
  Out[13]: [67.81, 81.32]

  In [14]: e6
  Out[14]: [78.09, 80.26]

  In [15]: import numpy as np

  In [16]: np.mean([ e1[0]-e1[1], e2[0]-e2[1], e3[0]-e3[1], e4[0]-e4[1], e5[0]-e5[1], e6[0]-e6[1] ])
  Out[16]: -6.086666666666663

And thus I'd say 6.1\%.
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
titlesize = 40
xsize = 40
ysize = 40
ticksize = 38
legendsize = 38
er_alpha = 0.25
bar_width = 0.3
bar_alpha = 0.9
lw = 7
EPS = 0.02
COLOR_RGB = 'red'
COLOR_DEP = 'blue'


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
    """Return a value between [0,100]."""
    if tier == 'Tier 1':
        return 95.7
    elif tier == 'Tier 2':
        return 94.6
    elif tier == 'Tier 3':
        return 95.1
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


def _debug(stats_bc, stats_d):
    # Print log for debugging / inspection.
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


def plot(files, all_rollout_data, rows_cols, success_thresh=0.92):
    """Plot rollout data, like other code except wrapping another list.

    all_rollout_data is a list where each item is a batch of rollout data (as
    in, another list) from a single training run. Thus, you can think of each
    item being one 'agent'.

    For each agent, I saved these keys: 'load_path', 'r', 'c', 'iv', 'oob',
    's_c', 's_iv'. Note that 's_c' = start coverage and 'c' = final coverage.
    """
    # I think it's easier to make the plot first, the iterate through the
    # agents and progressively add them. Note: adding 'wspace' and 'hspace'
    # keys to the gridspec_kw is tricky with tight_layout.
    nrows, ncols = 3, 2
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharey='row',
                           figsize=(11*ncols, 6*nrows),
                           gridspec_kw={'width_ratios':[1,1.45]})

    for idx, (file, agent_history, row_col) in enumerate(
                zip(files, all_rollout_data, rows_cols)):
        # Key assumption.
        is_rgb = idx < len(files) / 2

        # Actually we ignore the column here.
        rr, _ = row_col
        print('\nAt ({},{}), inspecting file {}, is RGB? {}'.format(rr,0,file,is_rgb))
        if len(agent_history) == 0:
            print('Agent has no history, skipping...')
            continue
        stats_bc = defaultdict(list)
        stats_d = defaultdict(list)
        current_tier = (agent_history[0])[-1] # last tuple item = tier
        print('it is: {}'.format(current_tier))

        for r_tuple in agent_history:
            # Each 'r_tuple' consists of one checkpoint's rolled out data.
            r_chkpt, r_data, r_file, r_tier = r_tuple
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

                # Eh I prefer splitting into bc vs dagger? For now.  See above
                # for `r_data[key]` details. I think we want to multiply by
                # 100, THEN take a standard deviation.
                if 'bc_' in r_chkpt:
                    final_cov = np.mean(100 * np.array(r_data[key]))
                    final_std = np.std(100 * np.array(r_data[key]))
                    stats_bc[key+'_mean'].append(final_cov)
                    stats_bc[key+'_std'].append(final_std)
                    if key == 'c':
                        success = np.sum(c >= success_thresh) / len(c)
                        stats_bc['success'].append( success )
                elif 'dagger_' in r_chkpt:
                    final_cov = np.mean(100 * np.array(r_data[key]))
                    final_std = np.std(100 * np.array(r_data[key]))
                    stats_d[key+'_mean'].append(final_cov)
                    stats_d[key+'_std'].append(final_std)
                    if key == 'c':
                        success = np.sum(c >= success_thresh) / len(c)
                        stats_d['success'].append( success )
                else:
                    raise ValueError(r_chkpt)

        # Pretty sure this will be ok.
        for key in stats_bc.keys():
            stats_bc[key] = np.array(stats_bc[key])
        for key in stats_d.keys():
            stats_d[key] = np.array(stats_d[key])
        _debug(stats_bc, stats_d)

        # Coverage.
        #ax[rr,0].set_title('Coverage'.format(), fontsize=titlesize)
        #ax[rr,0].set_ylabel('Coverage'.format(), fontsize=ysize)
        #ax[rr,0].set_ylim([0-EPS,1+EPS])

        ax[rr,0].set_title('Behavior Cloning, {}'.format(current_tier), fontsize=titlesize)
        ax[rr,0].set_ylabel('Final Coverage'.format(), fontsize=ysize)
        ax[rr,0].set_xlabel('Epochs'.format(), fontsize=xsize)
        ax[rr,0].set_ylim([0-EPS,100+EPS])

        ax[rr,1].set_title('DAgger, {}'.format(current_tier), fontsize=titlesize)
        #ax[rr,1].set_ylabel('Final Coverage'.format(), fontsize=ysize)
        ax[rr,1].set_xlabel('Iterations'.format(), fontsize=xsize)
        ax[rr,1].set_ylim([0-EPS,100+EPS])

        # Assume it's the last coordinate.
        def _get_xcoords(checkpoints):
            x = []
            for ch in checkpoints:
                split = ch.split('_')
                x.append( int(split[-1]) )
            return x

        # Plot the learned coverage and standard deviation.
        x_bc = _get_xcoords(stats_bc['chkpt'])
        x_d = _get_xcoords(stats_d['chkpt'])
        maxval_learner = max( np.max(stats_bc['c_mean']), np.max(stats_d['c_mean']) )
        if is_rgb:
            pol_label = 'Color (max: {:.1f})'.format(maxval_learner)
            pol_color = COLOR_RGB
        else:
            pol_label = 'Depth (max: {:.1f})'.format(maxval_learner)
            pol_color = COLOR_DEP
        ax[rr,0].plot(x_bc, stats_bc['c_mean'], lw=lw, marker='x', ms=18, mew=5,
                color=pol_color)
        ax[rr,1].plot(x_d, stats_d['c_mean'], lw=lw, marker='x', ms=18, mew=5,
                color=pol_color, label=pol_label)
        ax[rr,0].fill_between(x_bc,
                stats_bc['c_mean'] - stats_bc['c_std'],
                stats_bc['c_mean'] + stats_bc['c_std'],
                alpha=er_alpha)
        ax[rr,1].fill_between(x_d,
                stats_d['c_mean'] - stats_d['c_std'],
                stats_d['c_mean'] + stats_d['c_std'],
                alpha=er_alpha)

        # Only need this for one of RGB or depth. Note: no need to multiply by 100.
        if is_rgb:
            avg_start = np.mean(stats_bc['s_c_mean'])
            ax[rr,0].axhline(y=avg_start, ls='--', lw=5, color='gray',
                    label='Start: {:.1f}'.format(avg_start))
            ax[rr,1].axhline(y=avg_start, ls='--', lw=5, color='gray')
            avg_demo = _demo_coverage(current_tier)
            ax[rr,0].axhline(y=avg_demo, ls='--', lw=5, color='k',
                    label='Demo.: {:.1f}'.format(avg_demo))
            ax[rr,1].axhline(y=avg_demo, ls='--', lw=5, color='k')

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
    figname = 'fig_rollout_combo_v01.png'.format()
    plt.savefig(figname)
    print("Just saved: {}\n".format(figname))


if __name__ == "__main__":
    # a,b,c = tiers 1,2,3
    # rollouts_4 from rgb images
    # rollouts_6 from depth images
    # for simplicity please put in: color t1, color t2, color t3
    #                         then: depth t1, depth t2, depth t3
    # I assume later that it's ordered as: half color, half depth.
    HEAD = '/nfs/diskstation/seita/clothsim'
    FILES = [
        #join(HEAD,'rollouts_4.4.0a'),
        #join(HEAD,'rollouts_4.4.0b'), # I didn't run this experimental setting
        #join(HEAD,'rollouts_4.4.0c'),
        join(HEAD,'rollouts_4.5.0a'),
        join(HEAD,'rollouts_4.5.0b'),
        join(HEAD,'rollouts_4.5.0c'),
        #join(HEAD,'rollouts_6.1.0a'),
        #join(HEAD,'rollouts_6.1.0b'),
        #join(HEAD,'rollouts_6.1.0c'),
        join(HEAD,'rollouts_6.2.0a'),
        join(HEAD,'rollouts_6.2.0b'),
        join(HEAD,'rollouts_6.2.0c'),
    ]
    # The rows/cols of files above, make sure they match.
    ROW_COL = [
        (0,0),
        (1,0),
        (2,0),
        (0,0),
        (1,0),
        (2,0),
    ]
    ALL_ROLLOUT_DATA = []

    # Essentially the same as plot_rollouts except wrap another for loop.
    for f in FILES:
        print('\nCurrently on: {}'.format(f))
        files = [join(f,x) for x in os.listdir(f) if x[-4:] == '.pkl']

        if len(files) > 0:
            tier = _get_tier(files)

        rollout_data = []
        for f in files:
            with open(f, 'rb') as fh:
                data = pickle.load(fh)
                # Just to make it easier to sort by load path.  Also, apply desired
                # ordering for files to make life easier. Check that we loaded correctly.
                p_head, p_tail = os.path.split(data['load_path'])
                if 'bc_' not in p_tail:
                    p_tail = 'dagger_{}'.format(p_tail)
                rollout_data.append( (p_tail, data, f, tier) )

        for rd in rollout_data:
            pth, data, f, tier = rd
            print(' loaded: {}     from: {}'.format(pth, data['load_path']))
        ALL_ROLLOUT_DATA.append(rollout_data)

    assert len(FILES) == len(ALL_ROLLOUT_DATA) == len(ROW_COL)
    plot(FILES, ALL_ROLLOUT_DATA, ROW_COL)
