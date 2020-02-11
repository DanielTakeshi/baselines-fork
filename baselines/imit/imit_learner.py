import sys
import cv2
import functools
from functools import reduce
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from baselines import logger
from baselines.common.mpi_adam import MpiAdam
import baselines.common.tf_util as U
from baselines.common.tf_util import save_variables, load_variables
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


class IMIT(object):

    def __init__(self, *, actor, memory, observation_shape, action_shape, batch_size=128,
                 action_range=(-1., 1.), actor_l2_reg=0., actor_lr=1e-4, clip_norm=None,
                 use_keras=False):
        """IMIT.

        For images, cast to a new variable which gets cast to the float.  Then we let *that*
        (obs0_f_imgs) be input to subsequent code. NOTE THE DIVISION BY 255 HERE!!

        Alas, things do change if we are using keras or not ... because then we
        just get a feature at the end.
        """
        self.use_keras    = use_keras
        self.actions      = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')
        self.flag_teacher = tf.placeholder(tf.float32, shape=(None, 1), name='flag_teacher')

        # Set up the pre-trained model.
        if self.use_keras:
            raise NotImplementedError()
            #self.pretrained_model = ResNet50(weights='imagenet', include_top=False)
            # The model predictions before last layer result in a (?,7,7,2048)-sized tensor.
            #self.obs0_f_imgs = tf.placeholder(tf.float32, shape=(None,7,7,2048), name='obs0_f_imgs')
        else:
            self.obs0 = tf.placeholder(tf.int32, shape=(None,) + observation_shape, name='obs0_imgs')
            self.obs0_f_imgs = tf.cast(self.obs0, tf.float32) / 255.0

        # Parameters.
        self.memory = memory
        self.obs_shape = observation_shape
        self.action_range = action_range
        self.actor = actor
        self.actor_lr = actor_lr
        self.actor_l2_reg = actor_l2_reg
        self.clip_norm = clip_norm
        self.batch_size = batch_size

        # Create actor network, with observations as input.
        self.actor_tf = actor(self.obs0_f_imgs)
        self.setup_actor_optimizer()

    def setup_actor_optimizer(self):
        """Make actor loss, grads, and optimizer.

        For this it's pretty simple: a behavior cloning loss, using actor_tf
        for the current actor's output given the state, and actions as
        placeholder for what was sampled from the buffer.

        Apply on all samples, no filtering for teachers as in DDPG, which means
        we simply divide by the batch size. Or just do tf.reduce_mean, but be
        sure to do row-wise first (to get error  per sample in the batch) and
        then take the mean (to get average in the BATCH)!

        We apply L2 reg if desired (following DeepMind's DDPGfD). Careful
        w/variable names if we switch network construction code!!
        """
        logger.info('\nSetting up actor optimizer, with behavior cloning')
        self._diff_m = self.actor_tf - self.actions
        self._diff_v = tf.reduce_mean(tf.square(self._diff_m), axis=1, keepdims=True)
        self.bc_loss = tf.reduce_mean(self._diff_v)
        self.actor_loss = self.bc_loss
        logger.info('  diff_matrix: {}'.format(self._diff_m))
        logger.info('  diff_vector: {}'.format(self._diff_v))

        if self.actor_l2_reg > 0.:
            actor_reg_vars = [var for var in self.actor.trainable_vars if
                    ((var.name.endswith('/w:0') or var.name.endswith('/kernel:0')) and 'output' not in var.name)]
            for var in actor_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.actor_l2_reg))
            self.actor_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.actor_l2_reg),
                weights_list=actor_reg_vars
            )
            self.actor_loss += self.actor_reg

        actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}\n'.format(actor_nb_params))
        self.actor_grads = U.flatgrad(self.actor_loss, self.actor.trainable_vars, clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(var_list=self.actor.trainable_vars, beta1=0.9, beta2=0.999, epsilon=1e-08)

    def store_transition(self, obs0, action, is_teacher=False):
        """Store transitions for IMIT (only need obs0, action).

        Daniel: collected via VecEnv, so iterate through batch size and append
        individual components. It's serial but shouldn't be a time bottleneck.
        """
        B = obs0.shape[0]
        for b in range(B):
            obs = obs0[b]
            if not self.use_keras:
                obs = cv2.resize(obs, (self.obs_shape[0], self.obs_shape[1]))
            self.memory.append(obs, action[b], is_teacher=is_teacher)

    def train(self, return_l2=False):
        """Sample a batch, run tensorflow, update actor parameters.
        """
        batch = self.memory.sample(batch_size=self.batch_size)

        # Daniel: I have more items than needed, for debugging purposes.
        ops = [self.actor_grads, self.actor_loss, self.actor_reg, self.flag_teacher,
                self._diff_v, self.actor_tf, self.actions]

        # Form the feed-dict.
        feed = {}
        if self.use_keras:
            raise NotImplemetedError()
            # TODO: does not work! Says graph has been finalized.
            #x = batch['obs0']
            #x = preprocess_input(x)
            #features = self.pretrained_model.predict(x)
            #feed[self.obs0_f_imgs] = features
        else:
            feed[self.obs0] = batch['obs0']
        feed[self.actions] = batch['actions']
        feed[self.flag_teacher] =  batch['flag_teacher']

        actor_grads, actor_loss, actor_l2_loss, flag_t, diff_v, act_tf, act_ph = \
                self.sess.run(ops, feed_dict=feed)
        assert act_tf.shape == act_ph.shape and len(act_tf.shape) == 2
        assert diff_v.shape == flag_t.shape and len(diff_v.shape) == 2
        #print('\nact_tf {}:\n{}'.format(act_tf.shape, act_tf.T))
        #print('\nact_ph {}:\n{}'.format(act_ph.shape, act_ph.T))
        #print('\ndiff_v {}:\n{}'.format(diff_v.shape, diff_v.T))
        #print('\nflag_t {}:\n{}'.format(flag_t.shape, flag_t.T))

        self.actor_optimizer.update(actor_grads, stepsize=self.actor_lr)
        if return_l2:
            return actor_loss, actor_l2_loss
        else:
            return actor_loss

    def step(self, obs, apply_noise=False):
        """Apply the policy, no noise added.

        Returns a 4-tuple, only for compatibility with other code. We just care
        about returning the action as the first argument.

        Leaving apply_noise for compatibility with `baselines/run.py`.
        """
        if not self.use_keras:
            # obs is from env, which is (B,224,224,3), but for calling our
            # model, we want it (B,x,x,3) where x is our chosen smaller dimension
            obs_new = []
            for b in range(obs.shape[0]):
                resized = cv2.resize(obs[b], (self.obs_shape[0], self.obs_shape[1]))
                obs_new.append(resized)
            obs_new = np.array(obs_new)
            assert obs_new.shape == (obs.shape[0], self.obs_shape[0], self.obs_shape[1], 4)
            obs = obs_new

        actor_tf = self.actor_tf
        feed_dict = {self.obs0: U.adjust_shape(self.obs0, [obs])}
        action = self.sess.run(actor_tf, feed_dict=feed_dict)
        action = np.clip(action, self.action_range[0], self.action_range[1])
        return action, None, None, None

    def initialize(self, sess):
        # Daniel: following PPO2 code outline, hoping to save/load models.
        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)
        # Daniel: back to normal.
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.actor_optimizer.sync()
