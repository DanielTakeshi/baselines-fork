import tensorflow as tf
from baselines.common import tf_util
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.models import get_network_builder
import sys
import gym


class PolicyWithValue(object):
    """Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, latent, estimate_q=False, vf_latent=None,
                 sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask
        """
        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent
        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)

        # Daniel: I modified to also return the logstd, so we can print directly.
        self.pd, self.pi, self.logstd = \
                self.pdtype.pdfromlatent(latent, init_scale=0.01, return_logstd=True)

        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session()

        # Daniel: where we implement the value function head w/`fc` and 1 output node.
        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', env.action_space.n)
            self.vf = self.q
        else:
            self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:,0]

    def get_logstd_values(self):
        """Daniel: helps us inspect the logstd values for sanity checks.
        """
        return self.sess.run(self.logstd)

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, **extra_feed):
        """Compute next action(s) given the observation(s)

        Daniel: debug and look at actual mean vector (before noise) with:

            #print(self._evaluate([self.pi], observation, **extra_feed))
            #print(self.sess.run(self.pd.logstd)) # bad, no about obs placeholder!
            #print(self._evaluate([self.pd.logstd], observation, **extra_feed))

        We shouldn't need to pass an observation placeholder for the logstd. My
        guess is that the logstd (from `self.logstd`) is from a `tf.split()`
        operation with the mean as well, so because the mean (clearly) requires
        the observation as input, it forces the same for the logstd.

        Parameters:
        ----------
        observation     observation data (either single or a batch)
        **extra_feed    additional data such as state or mask (names of the
                        arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state,
            negative log likelihood of the action under current policy parameters) tuple
        """
        a, v, state, neglogp = self._evaluate(
                [self.action, self.vf, self.state, self.neglogp], observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)


def build_policy(env, policy_network, value_network=None,
                 normalize_observations=False, estimate_q=False,
                 limit_act_range=False, **policy_kwargs):
    """Daniel: builds the policy and value network.

    When calling `get_network_builder`, we look at the provided models, but
    these will give us 'latent' features. We can combine models together, and
    then if we do that (or if not) the last layer *before* the last dense layer
    is considered to be the 'latent' one. For example, if we call mlp, by
    default we get two hidden layers with tanh, so we have:

        input --> 64 --> tanh --> 64 --> tanh

    and then there is a 64 dimensional tensor (see `policy_latent`) that we
    pass as input to the next layer(s). If you're wondering how we get to the
    action dimension, look at `baselines.common.distributions` and see
    `pdfromlatent` methods. They do a final 'matching' dense layer, but with no
    activation by default. To avoid that, we need to form a new layer with the
    correct output dimensions based on the environment's action space.

    To debug network construction, use:
        tf_util.display_var_info(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    Tricky, returns the inner function here, which then makes the policy via
    the class above, which exposes `self.action`, `self.pi`, etc., to use.
    """
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space
        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)

        extra_tensors = {}
        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X
        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)
            if isinstance(policy_latent, tuple):
                policy_latent, recurrent_tensors = policy_latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)

        _v_net = value_network
        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                vf_latent = _v_net(encoded_x)

        # Daniel: doing this for action range.
        if limit_act_range:
            policy_latent = tf.nn.tanh(
                fc(policy_latent, 'pi', env.action_space.shape[0], init_scale=0.01, init_bias=0.0)
            )

        policy = PolicyWithValue(
            env=env,
            observations=X,
            latent=policy_latent,
            vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms
