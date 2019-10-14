import tensorflow as tf
import numpy as np
from baselines.common.models import get_network_builder
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
from baselines.common.mpi_running_mean_std import RunningMeanStd
import tensorflow.contrib.layers as layers


class Model(object):
    def __init__(self, name, network='mlp', **network_kwargs):
        self.name = name
        self.network = network
        if network != 'cloth_cnn':
            self.network_builder = get_network_builder(network)(**network_kwargs)

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.nb_actions = nb_actions

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if self.network == 'mlp':
                x = self.network_builder(obs)
                x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = tf.nn.tanh(x)
            elif self.network == 'cloth_cnn':
                # --------------------------------------------------------------
                # To make this a little easier, here we're just going to explicitly design our net.
                # ASSUME WE ALREADY DIVIDE BY 255 IN DDPG.PY, so obs uses tf.float32.
                # This design is up to debate. For example we really should crop the net.
                # We actually have a `cloth_cnn` but that's for PPO w/only states as input.
                # --------------------------------------------------------------
                activ = tf.nn.tanh
                h = obs
                with tf.variable_scope("convnet"):
                    h = activ(conv(h, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
                    h = activ(conv(h, 'c2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2)))
                    h = activ(conv(h, 'c3', nf=32, rf=3, stride=1, init_scale=np.sqrt(2)))
                    h = conv_to_fc(h)
                with tf.variable_scope("fcnet"):
                    h = activ(tf.layers.dense(h, 200, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='fc1'))
                    h = tf.layers.dense(h, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='fc2')
                    x = tf.nn.tanh(h) # restrict actions
            else:
                raise ValueError(self.network)
        return x


class Critic(Model):
    def __init__(self, name='critic', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.layer_norm = True

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if self.network == 'mlp':
                x = tf.concat([obs, action], axis=-1) # this assumes observation and action can be concatenated
                x = self.network_builder(x)
                x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='output')
            elif self.network == 'cloth_cnn':
                # --------------------------------------------------------------
                # To make this a little easier, here we're just going to explicitly design our net.
                # ASSUME WE ALREADY DIVIDE BY 255 IN DDPG.PY, so obs uses tf.float32.
                # This design is up to debate. For example we really should crop the net.
                # We actually have a `cloth_cnn` but that's for PPO w/only states as input.
                # --------------------------------------------------------------
                activ = tf.nn.tanh
                h = obs
                with tf.variable_scope("convnet"):
                    h = activ(conv(h, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
                    h = activ(conv(h, 'c2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2)))
                    h = activ(conv(h, 'c3', nf=32, rf=3, stride=1, init_scale=np.sqrt(2)))
                    h = conv_to_fc(h)
                with tf.variable_scope("fcnet"):
                    h = tf.concat([h, action], axis=-1) # this assumes observation and action can be concatenated
                    h = activ(tf.layers.dense(h, 200, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='fc1'))
                    x = tf.layers.dense(h, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='output')
            else:
                raise ValueError(self.network)
        return x
