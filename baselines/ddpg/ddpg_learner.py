from copy import copy
import functools
from functools import reduce
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from baselines import logger
from baselines.common.mpi_adam import MpiAdam
import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.tf_util import save_variables, load_variables
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / (stats.std + 1e-8)


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def get_target_updates(vars, target_vars, tau):
    logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
    assert len(actor.vars) == len(perturbed_actor.vars)
    assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)

    updates = []
    for var, perturbed_var in zip(actor.vars, perturbed_actor.vars):
        if var in actor.perturbable_vars:
            logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(actor.vars)
    return tf.group(*updates)


class DDPG(object):

    def __init__(self, actor, critic, memory, observation_shape, action_shape, param_noise=None, action_noise=None,
        gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=True,
        batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
        critic_l2_reg=0., actor_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.,
        bc_teacher_lambda=0.0, use_qfilter=False):
        """DDPG.

        I changed observation_range to (0, 255) for the image-based RL part
        because we don't divide our images by 255 until later. The action and
        return range should be OK.
        """
        # Inputs. Daniel: for images, cast to a new variable which gets cast to the float.
        # Assumes we detect via observation space; I think MuJoCo envs have obs shape length 1.
        # Then we let the remainder be input to subsequent code that uses observations.
        if len(observation_shape) > 1:
            self.obs0 = tf.placeholder(tf.int32, shape=(None,) + observation_shape, name='obs0_imgs')
            self.obs1 = tf.placeholder(tf.int32, shape=(None,) + observation_shape, name='obs1_imgs')
            self.obs0_f_imgs = tf.cast(self.obs0, tf.float32) / 255.0
            self.obs1_f_imgs = tf.cast(self.obs1, tf.float32) / 255.0
            assert not normalize_observations, 'Why normalize if we already divide by 255?'
            observation_range = (-np.inf, np.inf)  # We don't want to clip raw pixels here.
            self.use_images = True
            self.bc_teacher_lambda = bc_teacher_lambda
            self.use_qfilter = use_qfilter
        else:
            # Assuming default MuJoCo settings here.
            self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
            self.obs1 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs1')
            self.use_images = False
            self.bc_teacher_lambda = bc_teacher_lambda
            self.actor_l2_reg = 0.0
            self.use_qfilter = use_qfilter

        self.terminals1         = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards            = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.actions            = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')
        self.critic_target      = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')
        # Daniel: new for demos.
        self.flag_teacher       = tf.placeholder(tf.float32, shape=(None, 1), name='flag_teacher')

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.critic = critic
        self.actor = actor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg
        self.actor_l2_reg = actor_l2_reg

        # Observation normalization.
        if self.normalize_observations:
            with tf.variable_scope('obs_rms'):
                self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None

        # Daniel: this is where all the obs are subsequently passed, thus handle image case.
        # That way our feed_dicts in later code can still use self.{obs0,obs1}.
        if self.use_images:
            normalized_obs0 = tf.clip_by_value(normalize(self.obs0_f_imgs, self.obs_rms),
                self.observation_range[0], self.observation_range[1])
            normalized_obs1 = tf.clip_by_value(normalize(self.obs1_f_imgs, self.obs_rms),
                self.observation_range[0], self.observation_range[1])
        else:
            normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
                self.observation_range[0], self.observation_range[1])
            normalized_obs1 = tf.clip_by_value(normalize(self.obs1, self.obs_rms),
                self.observation_range[0], self.observation_range[1])

        # Return normalization.
        if self.normalize_returns:
            with tf.variable_scope('ret_rms'):
                self.ret_rms = RunningMeanStd()
        else:
            self.ret_rms = None

        # Create target networks.
        target_actor = copy(actor)
        target_actor.name = 'target_actor'
        self.target_actor = target_actor
        target_critic = copy(critic)
        target_critic.name = 'target_critic'
        self.target_critic = target_critic

        # Create networks and core TF parts that are shared across setup parts.
        # One actor. Two critics: action can be from:
        #   (1) itself (supplied via placeholder) -- for critic update, Q(s,a) sampled from RBuffer.
        #   (2) from actor_tf, supplied by the actor -- for actor update which maximizes Q(s,pi(o)).
        # Then create two de-normalized versions of those critics.
        #   self.critic_tf            : Q(s,a) where a is supplied by placeholder
        #   self.critic_with_actor_tf : Q(s,pi(s)) where pi(s) is the actor
        # Finally, get target Q values from target critic/actor.
        self.actor_tf = actor(normalized_obs0)
        self.normalized_critic_tf = critic(normalized_obs0, self.actions)
        self.critic_tf = denormalize(tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        self.normalized_critic_with_actor_tf = critic(normalized_obs0, self.actor_tf, reuse=True)
        self.critic_with_actor_tf = denormalize(tf.clip_by_value(self.normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        Q_obs1 = denormalize(target_critic(normalized_obs1, target_actor(normalized_obs1)), self.ret_rms)
        self.target_Q = self.rewards + (1. - self.terminals1) * gamma * Q_obs1

        # Daniel: add a Q-filter, 1 if Q(s,a) > Q(s,pi(s)) where former has `a` from *demonstrator*. Only after pre-training?
        self.flag_qfilter = tf.cast(self.critic_tf > self.critic_with_actor_tf, tf.float32)
        self.during_pretrain =  tf.placeholder(tf.float32, (), name="during_pretrain_flag")

        # Set up parts.
        if self.param_noise is not None:
            self.setup_param_noise(normalized_obs0)
        self.setup_actor_optimizer()
        self.setup_critic_optimizer()
        if self.normalize_returns and self.enable_popart:
            self.setup_popart()
        self.setup_stats()
        self.setup_target_network_updates()

        self.initial_state = None # recurrent architectures not supported yet

    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars, self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def setup_param_noise(self, normalized_obs0):
        assert self.param_noise is not None

        # Configure perturbed actor.
        param_noise_actor = copy(self.actor)
        param_noise_actor.name = 'param_noise_actor'
        self.perturbed_actor_tf = param_noise_actor(normalized_obs0)
        logger.info('setting up param noise')
        self.perturb_policy_ops = get_perturbed_actor_updates(self.actor, param_noise_actor, self.param_noise_stddev)

        # Configure separate copy for stddev adoption.
        adaptive_param_noise_actor = copy(self.actor)
        adaptive_param_noise_actor.name = 'adaptive_param_noise_actor'
        adaptive_actor_tf = adaptive_param_noise_actor(normalized_obs0)
        self.perturb_adaptive_policy_ops = get_perturbed_actor_updates(self.actor, adaptive_param_noise_actor, self.param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor_tf - adaptive_actor_tf)))

    def setup_actor_optimizer(self):
        """Make actor loss, grads, and optimizer. Several changes:

        We use a behavior cloning loss (with a Q-filter on top of that), using
        actor_tf for the current actor's output given the state, and actions as
        placeholder for what was sampled from the buffer. The latter might have
        student actions, in which case we ignore these w/the flag.

        We apply L2 reg if desired (following DeepMind's DDPGfD). Careful
        w/variable names if we switch network construction code!!

        (Nair et al., 2018) set the `bc_teacher_lambda` term I'm using to 1,
        and average out the BC loss by all items in the batch, *regardless* of
        whether the item passed the Q-filter or not. We're doing the same here
        by dividing by the sum of the number of teacher flags.
        """
        logger.info('\nsetting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)

        if self.bc_teacher_lambda > 0.:
            # Daniel: add Behavior cloning loss to the actor, but only on teacher samples!
            # I'm doing a reduce_sum and dividing by the total in the flag teacher.
            self._diff_m = self.actor_tf - self.actions
            self._diff_v = tf.reduce_mean(tf.square(self._diff_m), axis=1, keepdims=True)
            self._diff_f = self._diff_v * self.flag_teacher
            # Daniel: another idea is to apply q-filters only if we are past pre-training.
            if self.use_qfilter:
                logger.info('  applying Q-filter flag: {}'.format(self.flag_qfilter))
                self._diff_f = tf.cond(self.during_pretrain > 0.5,
                                       lambda: self._diff_f,                      # pretrain? identity
                                       lambda: self._diff_f * self.flag_qfilter)  # else? apply filter
            self.bc_loss = tf.reduce_sum(self._diff_f) / (tf.reduce_sum(self.flag_teacher) + 1e-6)
            self.actor_loss += self.bc_loss
            logger.info('  applying BC loss to actor with {}'.format(self.bc_teacher_lambda))
            logger.info('  diff_matrix: {}'.format(self._diff_m))
            logger.info('  diff_vector: {}'.format(self._diff_v))
            logger.info('  diff_filter: {}'.format(self._diff_f))

        if self.actor_l2_reg > 0.:
            actor_reg_vars = [var for var in self.actor.trainable_vars if
                    ((var.name.endswith('/w:0') or var.name.endswith('/kernel:0')) and 'output' not in var.name)]
            for var in actor_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.actor_l2_reg))
            actor_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.actor_l2_reg),
                weights_list=actor_reg_vars
            )
            self.actor_loss += actor_reg

        actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}\n'.format(actor_nb_params))
        self.actor_grads = U.flatgrad(self.actor_loss, self.actor.trainable_vars, clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(var_list=self.actor.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def setup_critic_optimizer(self):
        """Make critic loss, grads, and optimizer. Minor change w/L2 regularization.

        I didn't realize that our custom code would name the variables a bit different.
        It actually makes a huge difference, as the critic's default L2 is 0.01. Just be
        careful if we decide to re-name the variables or use a different TF construction.
        """
        logger.info('\nsetting up critic optimizer')
        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms), self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))

        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in self.critic.trainable_vars if
                    ((var.name.endswith('/w:0') or var.name.endswith('/kernel:0')) and 'output' not in var.name)]
            for var in critic_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg

        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}\n'.format(critic_nb_params))
        self.critic_grads = U.flatgrad(self.critic_loss, self.critic.trainable_vars, clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=self.critic.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def setup_popart(self):
        # See https://arxiv.org/pdf/1602.07714.pdf for details.
        self.old_std = tf.placeholder(tf.float32, shape=[1], name='old_std')
        new_std = self.ret_rms.std
        self.old_mean = tf.placeholder(tf.float32, shape=[1], name='old_mean')
        new_mean = self.ret_rms.mean

        self.renormalize_Q_outputs_op = []
        for vs in [self.critic.output_vars, self.target_critic.output_vars]:
            assert len(vs) == 2
            M, b = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1
            self.renormalize_Q_outputs_op += [M.assign(M * self.old_std / new_std)]
            self.renormalize_Q_outputs_op += [b.assign((b * self.old_std + self.old_mean - new_mean) / new_std)]

    def setup_stats(self):
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']
        ops += [reduce_std(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_std']

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['reference_action_mean']
        ops += [reduce_std(self.actor_tf)]
        names += ['reference_action_std']

        if self.param_noise:
            ops += [tf.reduce_mean(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_mean']
            ops += [reduce_std(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_std']

        self.stats_ops = ops
        self.stats_names = names

    def step(self, obs, apply_noise=True, compute_Q=True):
        """Apply the policy.

        Note the noise: for DDPG if we are *deploying* it, we should probably
        set the noise to False, such as for the `--play` option.
        """
        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actor_tf
        else:
            actor_tf = self.actor_tf
        feed_dict = {self.obs0: U.adjust_shape(self.obs0, [obs])}
        if compute_Q:
            action, q = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q = None

        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            #assert noise.shape == action[0].shape # daniel: with my fix, both are (numenv, acdim)
            assert noise.shape == action.shape, '{} {}'.format(noise.shape, action.shape)
            action += noise
        action = np.clip(action, self.action_range[0], self.action_range[1])

        return action, q, None, None

    def store_transition(self, obs0, action, reward, obs1, terminal1, is_teacher=False):
        """Store transitions for DDPG.

        Daniel: collected via VecEnv, so iterate through batch size and append
        individual components. It's serial but shouldn't be a time bottleneck.
        Note that all this seems to be done using one-step returns; I don't see
        n-step returns anywhere. Also, we should add an indication if this is a
        teacher sample.
        """
        reward *= self.reward_scale
        B = obs0.shape[0]
        for b in range(B):
            self.memory.append(obs0[b], action[b], reward[b], obs1[b], terminal1[b],
                               is_teacher=is_teacher)
            if self.normalize_observations:
                self.obs_rms.update(np.array([obs0[b]]))

    def train(self, during_pretrain=False):
        """Daniel: added during_pretrain in case we want to do anything different there.

        By default it's false (and float(during_pretrain)=0.0) to maintain backwards compatibility.
        """
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)

        if self.normalize_returns and self.enable_popart:
            old_mean, old_std, target_Q = \
                    self.sess.run([self.ret_rms.mean, self.ret_rms.std, self.target_Q], feed_dict={
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })
            self.ret_rms.update(target_Q.flatten())
            self.sess.run(self.renormalize_Q_outputs_op, feed_dict={
                self.old_std : np.array([old_std]),
                self.old_mean : np.array([old_mean]),
            })

            # Run sanity check. Disabled by default since it slows down things considerably.
            # print('running sanity check')
            # target_Q_new, new_mean, new_std = self.sess.run([self.target_Q, self.ret_rms.mean, self.ret_rms.std], feed_dict={
            #     self.obs1: batch['obs1'],
            #     self.rewards: batch['rewards'],
            #     self.terminals1: batch['terminals1'].astype('float32'),
            # })
            # print(target_Q_new, target_Q, new_mean, new_std)
            # assert (np.abs(target_Q - target_Q_new) < 1e-3).all()
        else:
            target_Q = self.sess.run(self.target_Q, feed_dict={
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })

        ## Daniel: use this for debugging extra DDPG features we implemented:
        #ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss,
        #        self.critic_tf, self.critic_with_actor_tf, self.flag_teacher,
        #        self.flag_qfilter, self._diff_f, self.actor_tf, self.actions]
        #actor_grads, actor_loss, critic_grads, critic_loss, Q_demo, Q_actor, flag_t, flag_q, diff_f, act_tf, act_ph = \
        #            self.sess.run(ops, feed_dict={
        #    self.obs0: batch['obs0'],
        #    self.actions: batch['actions'],
        #    self.critic_target: target_Q,
        #    self.flag_teacher: batch['flag_teacher'],
        #    self.during_pretrain: float(during_pretrain),
        #})
        #print('\nQ(s,a), Q(s,pi(s)), act_tf, act_ph, diff_f, flag_q, flag_t')
        #print(Q_demo.T)
        #print(Q_actor.T)
        #print('now actors:')
        #print(act_tf.T)
        #print(act_ph.T)
        #print('now diff/flags:')
        #print(diff_f.T)
        #print(flag_q.T)
        #print(flag_t.T)

        # Get all gradients and perform a synced update.
        ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss]
        actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, feed_dict={
            self.obs0: batch['obs0'],
            self.actions: batch['actions'],
            self.critic_target: target_Q,
            self.flag_teacher: batch['flag_teacher'],
            self.during_pretrain: float(during_pretrain),
        })
        self.actor_optimizer.update(actor_grads, stepsize=self.actor_lr)
        self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)

        return critic_loss, actor_loss

    def initialize(self, sess):
        # Daniel: following PPO2 code outline, hoping to save/load models.
        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)
        # Daniel: back to normal.
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.actor_optimizer.sync()
        self.critic_optimizer.sync()
        self.sess.run(self.target_init_updates)

    def update_target_net(self):
        self.sess.run(self.target_soft_updates)

    def get_stats(self):
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)
        values = self.sess.run(self.stats_ops, feed_dict={
            self.obs0: self.stats_sample['obs0'],
            self.actions: self.stats_sample['actions'],
        })

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        if self.param_noise is not None:
            stats = {**stats, **self.param_noise.get_stats()}

        return stats

    def adapt_param_noise(self):
        try:
            from mpi4py import MPI
        except ImportError:
            MPI = None

        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        batch = self.memory.sample(batch_size=self.batch_size)
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.obs0: batch['obs0'],
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        if MPI is not None:
            mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        else:
            mean_distance = distance

        if MPI is not None:
            mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        else:
            mean_distance = distance

        self.param_noise.adapt(mean_distance)
        return mean_distance

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            self.sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })
