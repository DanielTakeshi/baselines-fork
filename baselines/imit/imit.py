import os
import time
import sys
import os.path as osp
from collections import deque
import pickle
from baselines.imit.imit_learner import IMIT
from baselines.imit.memory import Memory
from baselines.imit.models import Actor
from baselines.ddpg.ddpg import _ddpg_demos
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


# Similar to gym_cloth's examples/analytic.py but somewhat cumbersome to merge.
class OracleCornerPolicy:

    def __init__(self):
        """Oracle corner based policy, cheating as we know the position of points.

        Note the targets, expressed as (x,y):
          upper right: (1,1)
          lower right: (1,0)
          lower left:  (0,0)
          upper left:  (0,1)
        The order in which we pull is important, though!  Choose the method to
        be rotation or distance-based. The latter seems to be more reasonable:
        pick the corner that is furthest from its target.

        Use `np.arctan2(deltay,deltax)` for angle in [-pi,pi] if we use angles.
        Be careful about the action parameterization and if we clip or not.  If
        clipping, we have to convert the x and y to each be in [0,1].

        For tier2 we may have different corner targets for a given point index.
        """
        pass

    def set_env_data_dummy(self, env):
        """A bit different from analytic, cheating a bit but w/e it's Python!
        """
        self._env = env
        self._init_type = env._init_type
        self._clip_act_space = env._clip_act_space
        self._delta_actions = env._delta_actions
        assert self._init_type in ['tier1','tier2','tier3'], self._init_type
        assert self._clip_act_space and self._delta_actions

    def set_env_data_subproc(self, init_type, clip_act_space, delta_actions):
        """Simliar to dummy method except we don't have access to env (I think).
        """
        self._init_type = init_type
        self._clip_act_space = clip_act_space
        self._delta_actions = delta_actions
        assert self._init_type in ['tier1','tier2','tier3'], self._init_type
        assert self._clip_act_space and self._delta_actions

    def get_info(self):
        info = {'init_type': self._init_type, 'clip_act_space': self._clip_act_space,
                'delta_actions': self._delta_actions}
        return info

    def _data_delta(self, pt, targx, targy, shrink=True):
        """Given pt and target locations, return info needed for action.

        Assumes DELTA actions. Returns x, y of the current point (which should
        be the target) but also the cx, and cy, which should be used if we are
        'clipping' it into [-1,1], but for the 80th time, this really means
        _expanding_ the x,y.
        """
        x, y = pt.x, pt.y
        cx = (x - 0.5) * 2.0
        cy = (y - 0.5) * 2.0
        dx = targx - x
        dy = targy - y
        dist = np.sqrt( (x-targx)**2 + (y-targy)**2 )
        # Sometimes we grab the top, and can 'over-pull' toward a background
        # corner. Thus we might as well try and reduce it a bit. Experiment!
        # ----------------------------------------------------------------------
        # YIKES/TODO: this is 0.95 but I did 0.9 for demonstrator data ...
        # change this to use 0.9? It shouldn't make too much of a difference,
        # but it is something to be aware of ...
        # ----------------------------------------------------------------------
        if shrink:
            dx *= 0.90
            dy *= 0.90
        return (x, y, cx, cy, dx, dy, dist)

    def get_action(self, cloth=None):
        """Analytic oracle corner policy. Two cases:

        If we don't pass in cloth, we assume we have set an `_env` variable
        with it. That _env will internally keep up to date with environment
        changes and thus we can query its pts, etc.

        Otherwise (for SubprocVecs) I pass in the cloth on the fly. I think
        because they are updated in separate processes, and we need to call
        them to get updated versions?
        """
        if cloth is None:
            cloth = self._env.cloth
        pts = cloth.pts
        assert len(pts) == 625, len(pts)
        if self._init_type == 'tier2' and (not cloth.init_side):
            self._ll = 576  # 600
            self._ul = 598  # 624
            self._lr = 26   # 0
            self._ur = 48   # 24
        else:
            self._ll = 26   # 0
            self._ul = 48   # 24
            self._lr = 576  # 600
            self._ur = 598  # 624
        x0, y0, cx0, cy0, dx0, dy0, dist0 = self._data_delta(pts[self._ur], targx=1, targy=1)
        x1, y1, cx1, cy1, dx1, dy1, dist1 = self._data_delta(pts[self._lr], targx=1, targy=0)
        x2, y2, cx2, cy2, dx2, dy2, dist2 = self._data_delta(pts[self._ll], targx=0, targy=0)
        x3, y3, cx3, cy3, dx3, dy3, dist3 = self._data_delta(pts[self._ul], targx=0, targy=1)
        maxdist = max([dist0, dist1, dist2, dist3])

        # Pick cloth corner furthest from the target.
        if dist0 == maxdist:
            x, y, cx, cy, dx, dy = x0, y0, cx0, cy0, dx0, dy0
        elif dist1 == maxdist:
            x, y, cx, cy, dx, dy = x1, y1, cx1, cy1, dx1, dy1
        elif dist2 == maxdist:
            x, y, cx, cy, dx, dy = x2, y2, cx2, cy2, dx2, dy2
        elif dist3 == maxdist:
            x, y, cx, cy, dx, dy = x3, y3, cx3, cy3, dx3, dy3

        action = [cx, cy, dx, dy]
        return action


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
          save_interval=5,  # reduce memory footprint
          bc_epochs=0,
          load_path=None,
          demos_path=None,
          bc_teacher_lambda=0.0,
          use_qfilter=False,
          **network_kwargs):
    """Learns policy using imitation (maybe DAgger) w/vectorized environments.

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

    # Daniel: NOTE/TODO testing, if I can.
    USE_KERAS = False
    # Daniel: should be False unless I'm doing some testing.
    do_valid_tests = False

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

    # Form XP (1M steps, same as in paper), and ONLY ACTOR net, no critic.
    # Daniel: force dtype here so we can use uint8 type images.
    assert env.observation_space.low.dtype == env.observation_space.high.dtype
    # Also changing to (100,100) unless we do keras/pretraining, to force smaller images.
    if USE_KERAS:
        obs_shape = env.observation_space.shape
    else:
        obs_shape = (100,100,3)

    memory = Memory(limit=int(rb_size),
                    action_shape=env.action_space.shape,
                    observation_shape=obs_shape,
                    dtype=env.observation_space.low.dtype,
                    do_valid=do_valid_tests)
    actor = Actor(nb_actions, network=network, use_keras=USE_KERAS, **network_kwargs)

    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))

    # The `learn` defaults above have priority over defaults in IMIT class.
    agent = IMIT(actor=actor,
                 memory=memory,
                 observation_shape=obs_shape,
                 action_shape=env.action_space.shape,
                 batch_size=batch_size,
                 actor_l2_reg=actor_l2_reg,
                 actor_lr=actor_lr,
                 use_keras=USE_KERAS)
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
        logger.info("\nInside IMIT, loading model from: {}".format(load_path))
        agent.load(load_path)
    # --------------------------------------------------------------------------
    sess.graph.finalize()

    # --------------------------------------------------------------------------
    # Daniel: populate replay buffer, followed by behavior cloning stage.
    # But if load_path is not None, then doesn't make sense -- we want to load.
    # We also don't need to do this if timesteps is 0 (e.g., for playing policy).
    # --------------------------------------------------------------------------
    # OK now we're assuming we do DAgger by default. If we don't want to do
    # DAgger (i.e., pure BC) then let's set time_steps=1 for training.
    # --------------------------------------------------------------------------
    if total_timesteps == 0:
        return agent
    assert seed == 1500, 'We normally want seed 1500, yet: {}'.format(seed)

    if (demos_path is not None and load_path is None):
        _ddpg_demos(demos_path, agent, memory, algo='IMIT')
        assert memory.nb_entries == memory.nb_teach_entries, memory.nb_entries
        if do_valid_tests:
            memory.set_valid_idx()
        checkdir = osp.join(logger.get_dir(), 'checkpoints')
        statsdir = osp.join(logger.get_dir(), 'pretrain_stats')
        os.makedirs(checkdir, exist_ok=True)
        os.makedirs(statsdir, exist_ok=True)

        # Pretrain, based on their training code for some # of minibatches.
        pt_actor_losses = []
        pt_actor_losses_l2 = []
        batches_per_ep = int(memory.nb_entries / batch_size)
        logger.info('Running BC for {} epochs'.format(bc_epochs))
        logger.info('  data size in memory: {}'.format(memory.nb_entries))
        logger.info('  each batch: {}, epoch mbs: {}'.format(batch_size, batches_per_ep))
        if do_valid_tests:
            logger.info('  memory valid idx: {}'.format(memory.nb_valid_items))
        pt_start = time.time()

        for epoch in range(1,bc_epochs+1):
            losses = [] # includes L2 fyi
            losses_l2 = []
            for _ in range(batches_per_ep):
                al, al_l2 = agent.train(return_l2=True)
                losses.append(al)
                losses_l2.append(al_l2)
            pt_actor_losses.append( np.mean(losses) )
            pt_actor_losses_l2.append( np.mean(losses_l2) )

            # Check and save model occasionally.
            if epoch == 1 or epoch % 10 == 0:
                pt_time = (time.time() - pt_start) / 60.
                logger.info('  epoch done: {}, loss over past epoch: {:.4f} (l2: {:.4f})'.format(
                        str(epoch).zfill(4), pt_actor_losses[-1], pt_actor_losses_l2[-1]))
                logger.info('  elapsed time: {:.1f}m'.format(pt_time))
                savepath = osp.join(checkdir, 'bc_epoch_{}'.format(str(epoch).zfill(4)))
                logger.info('Saving model checkpoint to: ', savepath)
                agent.save(savepath)

            # Do validation here.
            if do_valid_tests:
                num_mbs = int(memory.nb_valid_items / batch_size)
                l2_errors = []
                for mb in range(num_mbs):
                    res = memory.get_valid_obs(mb*batch_size, (mb+1)*batch_size)
                    valid_obs = res['obs0']
                    valid_act = res['actions']
                    assert valid_obs.shape == (batch_size,100,100,3), valid_obs.shape
                    acts, _, _, _ = agent.step(obs=valid_obs, apply_noise=False)
                    l2_err_vector = np.mean((valid_act - acts)**2, axis=1)
                    l2_errors.extend(l2_err_vector)

                # Last minibatch
                res = memory.get_valid_obs((mb+1)*batch_size, memory.nb_valid_items)
                valid_obs = res['obs0']
                valid_act = res['actions']
                acts, _, _, _ = agent.step(obs=valid_obs, apply_noise=False)
                l2_err_vector = np.mean((valid_act - acts)**2, axis=1)
                l2_errors.extend(l2_err_vector)

                l2_err_valid = np.mean(l2_errors)
                logger.log('  VALIDATION L2 error: {:.4f}'.format(l2_err_valid))

        pt_time = (time.time() - pt_start) / 60.
        logger.info('losses a: {}'.format(np.array(pt_actor_losses)))
        logger.info('losses a (l2 norm of weights): {}'.format(np.array(pt_actor_losses_l2)))
        losses_pth = osp.join(statsdir, 'bc_losses.pkl')
        losses_l2_pth = osp.join(statsdir, 'bc_losses_l2_only.pkl')
        with open(losses_pth, 'wb') as fh:
            pickle.dump(losses_pth, fh)
        with open(losses_l2_pth, 'wb') as fh:
            pickle.dump(losses_l2_pth, fh)
        logger.info('Finished BC (no DAgger) in {:.1f}m.\n'.format(pt_time))
    # --------------------------------------------------------------------------

    # Back to their code. For cloth, `env.reset()` takes a while so we put it here.
    obs = env.reset()
    if eval_env is not None:
        eval_obs = eval_env.reset()
    nenvs = obs.shape[0]

    # Daniel: this is how to get the cloth state for the teacher; depends on num_env.
    logger.info('nenvs={}, getting cloth points + teacher policy...'.format(nenvs))
    if nenvs == 1:
        # DummyVecEnv has an `envs` "list". Then an extra `.env` to get ClothEnv.
        cloth_env = (env.envs[0]).env
        pts = (cloth_env).cloth.pts
        logger.info('singleton env type: {}'.format(cloth_env))
        logger.info('len(points): {}'.format(len(pts)))
        teacher = OracleCornerPolicy()
        teacher.set_env_data_dummy(cloth_env)
        logger.info('teacher attributes: {}'.format(teacher.get_info()))
        teacher_list = [ teacher ]
    else:
        # SubprocVecEnv, not sure if we can obtain envs or if it's safe, so I did this.
        env_attr = env.get_cloth_attributes()
        logger.info('env attributes: {}'.format(env_attr))
        teacher_list = []
        assert len(env_attr) == nenvs, len(env_attr)
        for env_a in env_attr:
            teacher = OracleCornerPolicy()
            teacher.set_env_data_subproc(env_a[0], env_a[1], env_a[2])
            teacher_list.append(teacher)

    # Daniel: Debugging/sanity checks.
    _variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    U.display_var_info(_variables)
    logger.info("\nInside IMIT, about to start epochs")
    logger.info("nbatchsize: {}, get this in buffer before IMIT updates".format(nbatchsize))
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
    epoch_episodes = 0

    for epoch in range(nb_epochs):
        mb_actions = []
        mb_epinfos = []
        for cycle in range(nb_epoch_cycles):
            # Daniel: pure data collection (NO noise added) to populate replay buffer.
            # No training until after this, and note the parallel stepping (VecEnv).

            for t_rollout in range(nb_rollout_steps):
                # Predict next action: (#_parallel_envs, ac_dim).
                action, _, _, _ = agent.step(obs)
                if rank == 0 and render:
                    env.render()

                # Before environment stepping happens, we need to run the teacher
                # policy here, because it needs the SAME EXACT env state. The policy
                # does not apply on the obs but the INTERNAL env.cloth.pts.
                t_actions = []
                if nenvs > 1:
                    cloth_objs = env.get_cloth_objs()
                    for teacher, cloth in zip(teacher_list, cloth_objs):
                        t_act = teacher.get_action(cloth=cloth)
                        t_actions.append(t_act)
                else:
                    for teacher in teacher_list:
                        t_act = teacher.get_action()
                        t_actions.append(t_act)
                t_actions = np.array(t_actions)
                t_actions = np.maximum( np.minimum(t_actions,1.0), -1.0)
                logger.info('agent actions:\n{}'.format(action))
                logger.info('teacher actions:\n{}'.format(t_actions))
                logger.info('L2 diff: \n{:.4f}'.format(
                        np.mean( np.mean((action-t_actions)**2, axis=1) )
                ))

                # max_action is of dimension A, whereas action is dimension
                # (nenvs, A) - the multiplication gets broadcasted to the batch
                # scale for execution in env (as far as DDPG is concerned,
                # every action is in [-1, 1])
                new_obs, r, done, info = env.step(max_action * action)
                r = r.astype(np.float32)
                t += 1
                if rank == 0 and render:
                    env.render()
                episode_reward += r
                episode_step += 1
                epoch_actions.append(action)

                # Daniel: Same as PPO2/DDPG, just checking for end of episodes.
                mb_actions.append(action)
                for inf in info:
                    maybeepinfo = inf.get('episode')
                    if maybeepinfo:
                        mb_epinfos.append(inf)

                # The batched data will be unrolled in memory.py's append. Daniel:
                # unlike DDPG, we only need obs/act, but act is FROM THE EXPERT.
                # Unfortunately there will be duplicate obs/act pairs if the student
                # actions don't touch the cloth, but I'm not sure how to avoid that.
                agent.store_transition(obs, t_actions)
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

            # Train.
            epoch_actor_losses = []
            epoch_actor_losses_l2 = []
            for t_train in range(nb_train_steps):
                al, al_l2 = agent.train(return_l2=True)
                epoch_actor_losses.append(al)
                epoch_actor_losses_l2.append(al_l2)

        if MPI is not None:
            mpi_size = MPI.COMM_WORLD.Get_size()
        else:
            mpi_size = 1

        # Log stats.
        # XXX shouldn't call np.mean on variable length lists
        duration = time.time() - start_time
        combined_stats = {}
        combined_stats['memory/nb_entries'] = memory.nb_entries
        combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
        combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
        combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
        combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
        combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
        combined_stats['train/loss_actor_l2'] = np.mean(epoch_actor_losses_l2)
        combined_stats['total/duration'] = duration
        combined_stats['total/steps_per_second'] = float(t) / float(duration)
        combined_stats['total/episodes'] = episodes
        combined_stats['rollout/episodes'] = epoch_episodes
        combined_stats['rollout/actions_std'] = np.std(epoch_actions)

        def as_scalar(x):
            if isinstance(x, np.ndarray):
                assert x.size == 1
                return x[0]
            elif np.isscalar(x):
                return x
            else:
                raise ValueError('expected scalar, got %s'%x)

        # Total statistics.
        combined_stats_sums = np.array([ np.array(x).flatten()[0] for x in combined_stats.values()])
        if MPI is not None:
            combined_stats_sums = MPI.COMM_WORLD.allreduce(combined_stats_sums)
        combined_stats = {k : v / mpi_size for (k,v) in
                zip(combined_stats.keys(), combined_stats_sums)}
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

        # Daniel: we can use cycle or epoch for this if condition ... kind of annoying but w/e.
        if cycle % save_interval == 0:
            logger.info('We are now saving stuff!!')
            savepath = osp.join(checkdir, '%.5i'%epoch)
            logger.info('Saving model checkpoint to: ', savepath)
            agent.save(savepath)
            # ------------------------------------------------------------------
            # Daniel: extra stuff for debugging.
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
