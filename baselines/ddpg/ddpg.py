import os
import time
import sys
import os.path as osp
from collections import deque
import pickle
from baselines.ddpg.ddpg_learner import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines.common import set_global_seeds
import baselines.common.tf_util as U
from baselines import logger
import numpy as np
np.set_printoptions(suppress=True, edgeitems=40, linewidth=200)
import tensorflow as tf
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def _ddpg_demos(demos_path, agent, memory, algo='DDPG', lim_exp_samples=10000):
    """Use demonstrations with DDPG or IMIT.

    Assumes we populate the buffer first, and then we are done w/teacher
    samples. We can 'tell' the memory so it doesn't overwrite info.

    See `gym-cloth/analysis/check_demo_data.py` for how we might choose to
    filter our data to improve its quality. Unfortunately this is dependent on
    the reward type, and such, so double check always.
    """
    with open(demos_path, 'rb') as fh:
        demos = pickle.load(fh)
    logger.info('\nPre-training phase, {}, loaded {} demos.'.format(algo, len(demos)))
    all_l = []
    all_r = []
    use_r = []
    nb_ignore_oob = 0
    nb_ignore_cov = 0
    lim = 10            # TODO put in config?
    sparse_rew = False  # TODO put in config?

    for eidx,episode in enumerate(demos):
        all_l.append( len(episode['rew']) )
        all_r.extend( episode['rew'] )
        # Batch each episode's statistics.
        obs0 = np.array( episode['obs'] )[:-1]
        obs1 = np.array( episode['obs'] )[1:]
        act  = np.array( episode['act'] )
        rew  = np.array( episode['rew'] )
        done = np.array( episode['done'] )
        assert obs0.shape == obs1.shape
        assert obs0.shape[0] == act.shape[0] == rew.shape[0] == done.shape[0]
        if sparse_rew:
            # Then keep all states from all episodes, except for ignoring
            # entire episodes that did not succeed with > 0 reward.
            if sum(episode['rew']) < 1:
                num_ignore += 1
                continue
            use_r.extend( episode['rew'] )
        else:
            # From gym-cloth analysis, ignore (a) short episodes, (b) last few states.
            # Short episodes are bad + out of bounds, last few often very similar.
            # Update (August): actually for tier1 demos, keep all. For tier2,
            # filter by some out of bounds cases.
            lastinfo = episode['info'][-1]
            if lastinfo['out_of_bounds']:
                nb_ignore_oob += 1
                continue
            if lastinfo['actual_coverage'] < 0.9:
                nb_ignore_cov += 1
                continue
            use_r.extend( episode['rew'][:lim] )
            obs0 = obs0[:lim]
            obs1 = obs1[:lim]
            act  = act[:lim]
            rew  = rew[:lim]
            done = done[:lim]
        # Ah, sometimes demonstrator actions may actually be outside [-1,1].
        act = np.maximum(act, -1.)
        act = np.minimum(act,  1.)
        # Store them in the replay buffer. If IMIT, only need obs and act.
        if algo == 'IMIT':
            agent.store_transition(obs0, act, is_teacher=True)
        else:
            agent.store_transition(obs0, act, rew, obs1, terminal1=done, is_teacher=True)
        if memory.nb_entries > lim_exp_samples:
            logger.info('replay buffer with {} expert samples, exceeds {}'.format(
                    memory.nb_entries, lim_exp_samples))
            logger.info('  ... breaking at episode idx {}'.format(eidx))
            break

    # Set teacher idx so that we do not over-write teacher data.
    memory.set_teacher_idx()
    logger.info('  ignoring {} episodes, out of bounds'.format(nb_ignore_oob))
    logger.info('  ignoring {} episodes, coverage'.format(nb_ignore_cov))
    logger.info('Average demo ep length: {:.3f} +/- {:.1f}'.format(
            np.mean(all_l), np.std(all_l)))
    logger.info('Average demo rewards, all t: {:.3f} +/- {:.1f} ({} total)'.format(
            np.mean(all_r), np.std(all_r), len(all_r)))
    logger.info('Average rewards we use: {:.3f} +/- {:.1f} ({} total)'.format(
            np.mean(use_r), np.std(use_r), len(use_r)))
    logger.info('Memory size: {}, nb teacher samples: {}'.format(
            memory.nb_entries, memory.nb_teach_entries))
    logger.info('Done with {} demo insertion phase.\n'.format(algo))


def learn(network, env,
          seed=None,
          total_timesteps=None,
          nb_epochs=None, # with default settings, perform 1M steps total
          nb_epoch_cycles=20,
          nb_rollout_steps=100,
          reward_scale=1.0,
          render=False,
          render_eval=False,
          noise_type='adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=True,
          actor_l2_reg=0.0,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50, # per epoch cycle and MPI worker,
          nb_eval_steps=1000,
          batch_size=64, # per MPI worker
          tau=0.01,
          eval_env=None,
          param_noise_adaption_interval=50,
          rb_size=1e6,
          save_interval=1,
          pretrain_epochs=0,
          load_path=None,
          demos_path=None,
          bc_teacher_lambda=0.0,
          use_qfilter=False,
          **network_kwargs):
    """Learns policy using DDPG, with vectorized environments.

    If we pass other arguments that aren't specified here, they are considered
    as network_kwargs.

    Parameters
    ----------
    noise_type: for noise to be added to the behavior policy. They are NOT
        using the noise type from the paper but 'AdaptiveParamNoiseSpec'. I
        _think_ that if one does the OU process, we get action noise, but not
        parameter noise. Also, be sure to use `name_stdev` in that convention,
        as the code will split the argument at the underscores.

    actor_lr: 1e-4  (matches paper)

    critic_lr: 1e-3  (matches paper)

    critic_l2: 1e-2  (matches paper)

    gamma: 0.99  (matches paper)

    batch_size: 64  (matches paper for lower-dim env obs/states)

    tau: 0.01 for soft target updates of actor and critic nets. Paper used 0.001.

    nb_epoch_cycles: number of times we go through this cycle of: (1) get
        rollouts with noise added to policy and apply to replay buffer, (2)
        gradient updates for actor/critic, (3) evaluation rollouts (if any).
        AFTER all of these cycles happen, THEN we log statistics.

    nb_rollout_steps: number of steps in each parallel env we take with
        exploration policy without training, so this is just to populate the
        replay buffer. More parallel envs *should* mean that we get more
        samples in the buffer between each gradient updates of the network, so
        this might need to be environment *and* machine (# of CPUs) specific.

    nb_train_steps: after doing `nb_rollout_steps` in each parallel env, we do
        this many updates; each involves sampling from the replay buffer and
        updating the actor and critic (via lagged target updates).

    nb_eval_steps: 1000, I changed from the 100 as default. Using 1000 ensures
        that fixed length envs like Ant-v2 can get one full episode (assuming
        one parallel env) during evaluation stagtes.

    eval_env: A separate environment for evaluation only, where no noise is
        applied, similar to how rlkit does it.

    save_interval: Frequency between saving.
    """
    set_global_seeds(seed)

    # Daniel: this helps to maintain compatibility with PPO2 code. For now
    # we're ignoring it, but we should check that we're always clipping. I
    # changed the nb_epochs to match with PPO2 in that we divide by nenvs.
    if 'limit_act_range' in network_kwargs:
        network_kwargs.pop('limit_act_range')
    nenvs = env.num_envs

    nbatchsize = nenvs * nb_epoch_cycles * nb_rollout_steps
    if total_timesteps is not None:
        assert nb_epochs is None
        nb_epochs = int(total_timesteps) // nbatchsize
    else:
        nb_epochs = 500

    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0

    # we assume symmetric actions.
    nb_actions = env.action_space.shape[-1]
    assert (np.abs(env.action_space.low) == env.action_space.high).all()

    # Form XP (1M steps, same as in paper), and critic/actor networks.
    # Daniel: force dtype here so we can use uint8 type images.
    assert env.observation_space.low.dtype == env.observation_space.high.dtype
    memory = Memory(limit=int(rb_size),
                    action_shape=env.action_space.shape,
                    observation_shape=env.observation_space.shape,
                    dtype=env.observation_space.low.dtype)
    critic = Critic(network=network, **network_kwargs)
    actor = Actor(nb_actions, network=network, **network_kwargs)

    action_noise = None
    param_noise = None
    if noise_type is not None:
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev),
                                                     desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                #action_noise = NormalActionNoise(mu=np.zeros(nb_actions),
                #                                 sigma=float(stddev)*np.ones(nb_actions))
                #if nenvs > 1:
                # Daniel: adding this to replace the former.
                action_noise = NormalActionNoise(mu=np.zeros(nb_actions),
                                                 sigma=float(stddev)*np.ones(nb_actions),
                                                 shape=(nenvs,nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                            sigma=float(stddev)*np.ones(nb_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))

    # The `learn` defaults above have priority over defaults in DDPG class.
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns,
        normalize_observations=normalize_observations, batch_size=batch_size,
        action_noise=action_noise, param_noise=param_noise,
        actor_l2_reg=actor_l2_reg, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart,
        clip_norm=clip_norm, reward_scale=reward_scale,
        bc_teacher_lambda=bc_teacher_lambda, use_qfilter=use_qfilter)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Prepare everything.
    sess = U.get_session()
    agent.initialize(sess)
    # --------------------------------------------------------------------------
    # Daniel: similar as PPO2 code as `agent` is similar to `model` but has to
    # be initialized explicitly above. Must call after `agent.load` gets
    # created.  Not sure if this works with parameter space noise or with
    # normalization, but I don't plan to resume training (for now). It also has
    # to be *before* the `graph.finalize()` because otherwise we get an error.
    # --------------------------------------------------------------------------
    if load_path is not None:
        logger.info("\nInside ddpg, loading model from: {}".format(load_path))
        agent.load(load_path)
    # --------------------------------------------------------------------------
    sess.graph.finalize()
    agent.reset()

    # --------------------------------------------------------------------------
    # Daniel: populate replay buffer, followed by pretraining stage.
    # But if load_path is not None, then doesn't make sense -- we want to load.
    # We also don't need to do this if timesteps is 0 (e.g., for playing policy).
    # --------------------------------------------------------------------------
    if total_timesteps == 0:
        return agent
    assert seed == 1500, 'We normally want seed 1500, yet: {}'.format(seed)

    if (demos_path is not None and load_path is None):
        _ddpg_demos(demos_path, agent, memory)
        assert memory.nb_entries == memory.nb_teach_entries, memory.nb_entries
        checkdir = osp.join(logger.get_dir(), 'checkpoints')
        os.makedirs(checkdir, exist_ok=True)

        # Pretrain, based on their training code for some # of minibatches.
        pt_actor_losses = []
        pt_critic_losses = []
        batches_per_ep = int(memory.nb_entries / batch_size)
        logger.info('Running pre-training for {} epochs'.format(pretrain_epochs))
        logger.info('  data size in memory: {}'.format(memory.nb_entries))
        logger.info('  each batch: {}, epoch mbs: {}'.format(batch_size, batches_per_ep))
        pt_start = time.time()

        for epoch in range(1,pretrain_epochs+1):
            c_losses = []
            a_losses = []
            for _ in range(batches_per_ep):
                cl, al = agent.train(during_pretrain=True)
                agent.update_target_net()
                c_losses.append(cl)
                a_losses.append(al)
            pt_critic_losses.append( np.mean(c_losses) )
            pt_actor_losses.append( np.mean(a_losses) )

            # Check and save model occasionally.
            if epoch == 1 or epoch % 5 == 0:
                pt_time = (time.time() - pt_start) / 60.
                logger.info('  epoch done: {}, loss over past epoch: {:.4f}'.format(
                        str(epoch).zfill(4), pt_actor_losses[-1]))
                logger.info('  critic loss over past epoch: {:.4f}'.format(
                        pt_critic_losses[-1]))
                logger.info('  elapsed time: {:.1f}m'.format(pt_time))
                savepath = osp.join(checkdir, 'pretrain_epoch_{}'.format(str(epoch).zfill(4)))
                logger.info('Saving model checkpoint to: ', savepath)
                agent.save(savepath)

        pt_time = (time.time() - pt_start) / 60.
        logger.info('losses a: {}'.format(np.array(pt_actor_losses)))
        logger.info('losses c: {}'.format(np.array(pt_critic_losses)))
        logger.info('Finished loading teacher samples + pre-training.')
        logger.info('Pre-training took {:.1f}m.\n'.format(pt_time))
    # --------------------------------------------------------------------------

    # Back to their code. For cloth, `env.reset()` takes a while so we put it here.
    obs = env.reset()
    if eval_env is not None:
        eval_obs = eval_env.reset()
    nenvs = obs.shape[0]

    # Daniel: Debugging/sanity checks.
    _variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    U.display_var_info(_variables)
    logger.info("\nInside DDPG, about to start epochs")
    logger.info("nbatchsize: {}, get this in buffer before DDPG updates".format(nbatchsize))
    logger.info("  i.e.: (nenv {}) * (cycles {}) * (nsteps {})".format(
            nenvs, nb_epoch_cycles, nb_rollout_steps))
    logger.info("nb_epochs: {}, number of cycles to use".format(nb_epochs))
    logger.info("eval_env None? {}".format(eval_env is None))
    logger.info("(end of debugging messages)\n")

    # File paths.
    checkdir    = osp.join(logger.get_dir(), 'checkpoints')
    action_dir  = osp.join(logger.get_dir(), 'actions')
    episode_dir = osp.join(logger.get_dir(), 'ep_all_infos')
    os.makedirs(checkdir, exist_ok=True)
    os.makedirs(action_dir, exist_ok=True)
    os.makedirs(episode_dir, exist_ok=True)

    # Daniel: use these two to store past 100 episode history. Report these stats!
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    all_eval_episode_rewards = []

    # reward/step: cumulative quantities for each episode in vecenv.
    # epoch_{actions,qs} will grow without bound, fyi.
    episode_reward = np.zeros(nenvs, dtype = np.float32) #vector
    episode_step = np.zeros(nenvs, dtype = int) # vector
    episodes = 0 #scalar
    t = 0 # scalar
    epoch = 0
    start_time = time.time()
    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_actions = []
    epoch_qs = []
    epoch_episodes = 0

    for epoch in range(nb_epochs):
        mb_actions = []
        mb_epinfos = []
        for cycle in range(nb_epoch_cycles):
            # Perform rollouts.
            if nenvs > 1:
                # if simulating multiple envs in parallel, impossible to reset
                # agent at the end of the episode in each of the environments,
                # so resetting here instead
                agent.reset()

            # Daniel: pure data collection (noise added) to populate replay buffer.
            # No training until after this, and note the parallel stepping (VecEnv).
            for t_rollout in range(nb_rollout_steps):
                # Predict next action.
                # action: (#_parallel_envs, ac_dim), q: (#_parallel_envs, 1)
                action, q, _, _ = agent.step(obs, apply_noise=True, compute_Q=True)
                # Execute next action.
                if rank == 0 and render:
                    env.render()

                # max_action is of dimension A, whereas action is dimension
                # (nenvs, A) - the multiplication gets broadcasted to the batch
                # scale for execution in env (as far as DDPG is concerned,
                # every action is in [-1, 1])
                new_obs, r, done, info = env.step(max_action * action)
                r = r.astype(np.float32)
                # note these outputs are batched from vecenv (first dim = batch).
                t += 1
                if rank == 0 and render:
                    env.render()
                episode_reward += r
                episode_step += 1

                # Book-keeping. (Daniel: agent.train() doesn't require these two lists)
                epoch_actions.append(action)
                epoch_qs.append(q)
                # Daniel: Same as PPO2 code.
                mb_actions.append(action)
                for inf in info:
                    maybeepinfo = inf.get('episode')
                    if maybeepinfo:
                        mb_epinfos.append(inf)
                #the batched data will be unrolled in memory.py's append.
                agent.store_transition(obs, action, r, new_obs, done)
                obs = new_obs

                for d in range(len(done)):
                    if done[d]:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward[d])    # Entire history
                        episode_rewards_history.append(episode_reward[d])  # Last 100 only
                        epoch_episode_steps.append(episode_step[d])
                        episode_reward[d] = 0.
                        episode_step[d] = 0
                        epoch_episodes += 1
                        episodes += 1
                        if nenvs == 1:
                            agent.reset()

            # Train.
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []
            for t_train in range(nb_train_steps):
                # Adapt param noise, if necessary.
                if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                    distance = agent.adapt_param_noise()
                    epoch_adaptive_distances.append(distance)

                cl, al = agent.train()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.update_target_net()

            # Evaluate. (Daniel: note that no noise is applied here.)
            # Also it seems like episodes do not naturally reset before this starts?
            # Also, unlike epoch_episode_reward, here we create eval_episode_reward here ...
            eval_episode_rewards = []
            eval_qs = []
            if eval_env is not None:
                logger.info('Now on the eval_env for {} steps...'.format(nb_eval_steps))
                nenvs_eval = eval_obs.shape[0]
                eval_episode_reward = np.zeros(nenvs_eval, dtype = np.float32)
                for t_rollout in range(nb_eval_steps):
                    eval_action, eval_q, _, _ = agent.step(eval_obs, apply_noise=False, compute_Q=True)
                    # scale for execution in env (for DDPG, every action is in [-1, 1])
                    eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)
                    if render_eval:
                        eval_env.render()
                    eval_episode_reward += eval_r

                    eval_qs.append(eval_q)
                    for d in range(len(eval_done)):
                        if eval_done[d]:
                            eval_episode_rewards.append(eval_episode_reward[d])
                            eval_episode_rewards_history.append(eval_episode_reward[d])
                            all_eval_episode_rewards.append(eval_episode_reward[d])
                            eval_episode_reward[d] = 0.0  # Daniel: reset for next episode.

        if MPI is not None:
            mpi_size = MPI.COMM_WORLD.Get_size()
        else:
            mpi_size = 1

        # Log stats.
        # XXX shouldn't call np.mean on variable length lists
        duration = time.time() - start_time
        stats = agent.get_stats()
        combined_stats = stats.copy()
        combined_stats['memory/nb_entries'] = memory.nb_entries
        combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
        combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
        combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
        combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
        combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
        combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
        combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
        combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
        combined_stats['total/duration'] = duration
        combined_stats['total/steps_per_second'] = float(t) / float(duration)
        combined_stats['total/episodes'] = episodes
        combined_stats['rollout/episodes'] = epoch_episodes
        combined_stats['rollout/actions_std'] = np.std(epoch_actions)
        # Evaluation statistics. (Daniel: use eval/return_history for plots)
        if eval_env is not None:
            combined_stats['eval/return'] = np.mean(eval_episode_rewards)
            combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
            combined_stats['eval/Q'] = eval_qs
            combined_stats['eval/episodes'] = len(eval_episode_rewards)
        def as_scalar(x):
            if isinstance(x, np.ndarray):
                assert x.size == 1
                return x[0]
            elif np.isscalar(x):
                return x
            else:
                raise ValueError('expected scalar, got %s'%x)

        combined_stats_sums = np.array([ np.array(x).flatten()[0] for x in combined_stats.values()])
        if MPI is not None:
            combined_stats_sums = MPI.COMM_WORLD.allreduce(combined_stats_sums)

        combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

        # Total statistics.
        combined_stats['total/epochs'] = epoch + 1
        combined_stats['total/steps_per_env'] = t

        for key in sorted(combined_stats.keys()):
            logger.record_tabular(key, combined_stats[key])

        if rank == 0:
            logger.dump_tabular()
        logger.info('')
        logdir = logger.get_dir()
        if rank == 0 and logdir:
            if hasattr(env, 'get_state'):
                with open(osp.join(logdir, 'env_state.pkl'), 'wb') as f:
                    pickle.dump(env.get_state(), f)
            if eval_env and hasattr(eval_env, 'get_state'):
                with open(osp.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                    pickle.dump(eval_env.get_state(), f)
            # Daniel: arguable, we can save all episodes but hard if we don't know the steps.
            #if eval_env:
            #    with open(os.path.join(logdir, 'all_eval_episode_rewards.pkl'), 'wb') as f:
            #        pickle.dump(all_eval_episode_rewards, f)

        # Daniel: we can use cycle or epoch for this if condition ... kind of annoying but w/e.
        if cycle % save_interval == 0:
            logger.info('We are now saving stuff!!')
            savepath = osp.join(checkdir, '%.5i'%epoch)
            logger.info('Saving model checkpoint to: ', savepath)
            agent.save(savepath)
            # ------------------------------------------------------------------
            # Daniel: extra stuff for debugging PPO on cloth, actions and infos for each episode.
            mb_actions = _sf01(np.asarray(mb_actions))
            act_savepath = osp.join(action_dir, 'actions_%.5i.pkl'%epoch)
            epi_savepath = osp.join(episode_dir, 'infos_%.5i.pkl'%epoch)
            with open(act_savepath, 'wb') as fh:
                pickle.dump(mb_actions, fh)
            with open(epi_savepath, 'wb') as fh:
                pickle.dump(mb_epinfos, fh)

        # Daniel: we were not resetting earlier. Actually there are other
        # epoch_stats which we might consider resetting here?
        epoch_episodes = 0

    return agent


# See PPO2 runner for swapping axes.
def _sf01(arr):
    """swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
