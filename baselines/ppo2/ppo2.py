import pickle
import os
import sys
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.policies import build_policy
from baselines.common import tf_util
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.ppo2.runner import Runner
import tensorflow as tf


def constfn(val):
    def f(_):
        return val
    return f


def learn(*, network, env, total_timesteps, eval_env = None,
            seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, model_fn=None, **network_kwargs):
    '''Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.

    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)
                                      Daniel: should be `T` in the paper. Atari defaults are 128 as in the paper.

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective
                                      Daniel: 0.5 by default but the PPO paper uses 1.0 for Atari games.

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update
                                      Daniel: 4 by default but the PPO paper uses 3 (for Atari games), etc.

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training. Daniel: for `tf.clip_by_value(ratio, 1-cliprange, 1+cliprange)`

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.
    '''
    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    # Daniel: hacky within PPO2 solution for limiting action ranges.
    if 'limit_act_range' in network_kwargs:
        limit_act_range = network_kwargs['limit_act_range']
        network_kwargs.pop('limit_act_range')
    else:
        limit_act_range = False

    policy = build_policy(env, network, limit_act_range=limit_act_range, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.ppo2.model import Model
        model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space,
                     nbatch_act=nenvs, nbatch_train=nbatch_train,
                     nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                     max_grad_norm=max_grad_norm)

    if load_path is not None:
        logger.info("\nInside ppo2, loading model from: {}".format(load_path))
        model.load(load_path)

    # Daniel: debugging and sanity checks
    _variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    tf_util.display_var_info(_variables)

    # Instantiate the runner object (Daniel: calls `env.reset()` so can take a while for cloth)
    # Also, I'm going to assume that if total_timesteps=0 then we don't waste time creating this.
    if total_timesteps > 0:
        runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    # Start total timer
    tfirststart = time.perf_counter()
    nupdates = total_timesteps//nbatch

    # Daniel: debugging and sanity checks
    logger.info("\nInside ppo2, before updates (`env.reset()` called before this)")
    logger.info("  nsteps: {}, each env in VecEnv does this many to get minibatch".format(nsteps))
    logger.info("  nbatch: {}, i.e., nsteps * nenv, size of data from (get_minibatch)".format(nbatch))
    logger.info("  nbatch_train: {}, batch size for actual gradient update within epoch".format(nbatch_train))
    logger.info("  noptepochs: {}, number of epochs over collected minibatch for PPO updates".format(noptepochs))
    logger.info("  nupdates: {}, number of (get_minibatch, update_net) cycles".format(nupdates))
    logger.info("  our model_fn class: {}".format(model_fn))
    logger.info("(end of debugging messages)\n")

    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)
        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, states, epinfos, ep_all_infos = runner.run() #pylint: disable=E0632
        if eval_env is not None:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos, eval_ep_all_infos = eval_runner.run() #pylint: disable=E0632

        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and (MPI is None or MPI.COMM_WORLD.Get_rank() == 0):
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            logger.info('Saving model checkpoint to: ', savepath)
            model.save(savepath)
            # ------------------------------------------------------------------
            # Daniel: extra stuff for debugging PPO on cloth, actions and infos for each episode.
            logstd_vals = model.act_model.get_logstd_values()
            action_dir  = osp.join(logger.get_dir(), 'actions')
            episode_dir = osp.join(logger.get_dir(), 'ep_all_infos')
            logstd_dir  = osp.join(logger.get_dir(), 'logstd')
            os.makedirs(action_dir, exist_ok=True)
            os.makedirs(episode_dir, exist_ok=True)
            os.makedirs(logstd_dir, exist_ok=True)
            act_savepath = osp.join(action_dir, 'actions_%.5i.pkl'%update)
            epi_savepath = osp.join(episode_dir, 'infos_%.5i.pkl'%update)
            std_savepath = osp.join(logstd_dir, 'logstd_%.5i.pkl'%update)
            with open(act_savepath, 'wb') as fh:
                pickle.dump(actions, fh)
            with open(epi_savepath, 'wb') as fh:
                pickle.dump(ep_all_infos, fh)
            with open(std_savepath, 'wb') as fh:
                pickle.dump(logstd_vals, fh)
            # ------------------------------------------------------------------
    return model


# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
