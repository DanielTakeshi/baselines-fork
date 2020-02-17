import sys
import os
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np
import yaml
import datetime
import pickle
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None
try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    """Daniel: starts training procedure, do bunch of checks for cloth env.

    The `model = learn(...)` is what calls algorithm-specific training code.
    Our cloth_config path should be in `extra_args` for now, and we can use
    that config for the random seed. This only applies if we didn't put in a
    `-seed` argument on the command line.

    We ALSO need that config path as input to even create the ClothEnv, and
    other arguments. Put those in `extra_args`.
    """
    env_type, env_id = get_env_type(args)
    logger.log('env_type: {}, env_id: {}'.format(env_type, env_id))
    extra_data = None  # Daniel: added this

    # Daniel: we have to figure out a cleaner way
    cloth_cfg_path = extra_args['cloth_config'] if 'cloth_config' in extra_args else None
    render_path = extra_args['render_path'] if 'render_path' in extra_args else None
    init_state = extra_args['init_state'] if 'init_state' in extra_args else None
    demos_path = extra_args['demos_path'] if 'demos_path' in extra_args else None

    if env_type == 'cloth':
        with open(cloth_cfg_path, 'r') as fh:
            cloth_config = yaml.safe_load(fh)
            extra_data = cloth_config
            if args.seed is None:
                args.seed = cloth_config['seed']
            if 'clip_act_space' in cloth_config['env']:
                extra_args['limit_act_range'] = cloth_config['env']['clip_act_space']

        # Force us to state/restrict network design to help our sanity.
        if cloth_config['env']['obs_type'] == '1d':
            if args.network is None:
                args.network = 'mlp'
            else:
                assert args.network == 'mlp', args.network
        elif cloth_config['env']['obs_type'] == 'blender':
            if args.network is None:
                args.network = 'cloth_cnn'
            else:
                assert args.network == 'cloth_cnn', args.network
        else:
            raise ValueError(cloth_config['env']['obs_type'])

        # Save the cloth file in case we want to verify the settings we used.
        _cloth_path = osp.join(logger.get_dir(), os.path.basename(cloth_cfg_path))
        with open(_cloth_path, 'w') as fh:
            yaml.dump(cloth_config, fh, default_flow_style=False)

        # Remove stuff because other dicts don't want unexpected args when we make network
        extra_args.pop('cloth_config', None)
        extra_args.pop('render_path', None)
        extra_args.pop('init_state', None)
        #extra_args.pop('demos_path', None) Actually we'd like to pass this in.
    else:
        assert 'cloth_config' not in extra_args
        assert 'render_path' not in extra_args
        assert 'init_state' not in extra_args
        assert 'demos_path' not in extra_args

    total_timesteps = int(args.num_timesteps)
    seed = args.seed
    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args,
                    cloth_cfg_path=cloth_cfg_path,
                    render_path=render_path,
                    start_state_path=init_state)

    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"),
                               record_video_trigger=lambda x: x % args.save_video_interval == 0,
                               video_length=args.save_video_length)
    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    # Debugging paths because we have to change paths for rendering the cloth.
    logger.log('\nTraining {} on {}:{} with arguments \n{}'.format(
            args.alg, env_type, env_id, alg_kwargs))
    logger.log('path: {}'.format(os.path.dirname(__file__)))
    logger.log('working dir: {}\n'.format(os.getcwd()))

    # Daniel: new, don't forget to build an evaluation env without noise applied.
    if args.alg == 'ddpg' or args.alg == 'imit':
        # NOTE: this will cause duplicate print logs to appear. It SHOULD be OK though!
        #eval_env = build_env(args,
        #                     cloth_cfg_path=cloth_cfg_path,
        #                     render_path=render_path,
        #                     start_state_path=init_state)
        eval_env = None
        # TODO: check that adding eval env doesn't affect monitor.csv files.
        model = learn(
            env=env,
            eval_env=eval_env,
            seed=seed,
            total_timesteps=total_timesteps,
            **alg_kwargs
        )
    else:
        model = learn(
            env=env,
            seed=seed,
            total_timesteps=total_timesteps,
            **alg_kwargs
        )

    return model, env, extra_data


def build_env(args, cloth_cfg_path=None, render_path=None, start_state_path=None):
    """Daniel: actually construct the env, using 'vector envs' for parallelism.

    For now our cloth env can follow the non-atari and non-retro stuff, because
    I don't think we need a similar kind of 'wrapping' that they do. Note that
    `VecFrameStack` is needed to stack frames, e.g., in Atari we do 4 frame
    stacking. Without that, the states would be size (84,84,1).

    The non-`args` parameters here are for the cloth env.
    """
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed
    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed,
                               gamestate=args.gamestate,
                               reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)
    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)
        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed,
                           reward_scale=args.reward_scale,
                           flatten_dict_observations=flatten_dict_observations,
                           cloth_cfg_path=cloth_cfg_path,
                           render_path=render_path,
                           start_state_path=start_state_path)
        # https://github.com/openai/baselines/issues/938
        if env_type == 'mujoco' and alg != 'ddpg':
            env = VecNormalize(env)

    return env


def get_env_type(args):
    """Daniel: deduces environment type and ID.

    - env_type is the class (e.g., 'atari')
    - env_id is the actual title (e.g., 'PongNoFrameskip-v4').

    We have to make some changes to support Cloth-v0, our custom env.  Note
    that env_type is used to get defaults for network architectures, etc.
    """
    env_id = args.env

    if env_id == 'Cloth-v0':
        return 'cloth', env_id

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    """Daniel: also works for cloth, just define a cloth() method in the alg module.
    """
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary,
    evaluating python objects when possible
    '''
    def parse(v):
        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    # Daniel: let's make it clear that if `--play` is enabled, we shouldn't be training.
    if args.play:
        assert args.num_timesteps == 0, \
            'You should not call `--play` with more than one time steps.'

    # Daniel: just add `args.save_path` for putting logs somewhere reasonable.
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure(dir=args.save_path)
    else:
        logger.configure(dir=args.save_path, format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    # Daniel: made cloth_config_path a return value.
    model, env, cloth_config = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    # We support parallelism so might as well increase number of episodes.
    NB_EVAL_EPIS = 50

    # Subject to change depending on our reward threshold / design.
    success_threshold = 3.0

    if args.play:
        logger.info("\nRunning trained model -- no more training is happening.")
        logger.info("Will run for {} episodes.\n".format(NB_EVAL_EPIS))

        # Some checks to ensure correct parameters. Comment out if not desired.
        if args.env == 'Cloth-v0':
            test_seed = 1600
            if args.seed is None:
                assert cloth_config['seed'] == test_seed, cloth_config['seed']
            else:
                assert args.seed == test_seed, args.seed
            assert cloth_config['env']['force_grab'], 'Usually you want force_grab==True'

        obs = env.reset()
        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))
        episode_rews = np.zeros((args.num_env,))
        pstats = defaultdict(list)
        pstats['load_path'] = extra_args['load_path']
        logger.info('obs shape: {}'.format(obs.shape))

        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                # Daniel: DDPG/IMIT case happens here, so we should prob set noise=False.
                actions, _, _, _ = model.step(obs, apply_noise=False)
            logger.info('executing actions:\n{}'.format(actions))

            obs, rew, done, epinfo  = env.step(actions)
            #episode_rew += rew[0] if isinstance(env, VecEnv) else rew
            episode_rews += rew

            # Daniel: don't render the cloth, we have our own way of doing it.
            if args.env != 'Cloth-v0':
                env.render()

            done_boolean = done.any() if isinstance(done, np.ndarray) else done
            if done_boolean:
                logger.info('\n\n ************ FINISHED EPISODE, done: {} ************'.format(done))
                logger.info('episode_rews: {}'.format(episode_rews))
                logger.info('And epinfo: {} (len {})'.format(epinfo, len(epinfo)))
                for d in range(args.num_env):
                    if done[d]:
                        pstats['r'].append(episode_rews[d])
                        pstats['c'].append(epinfo[d]['actual_coverage'])
                        pstats['iv'].append(epinfo[d]['variance_inv'])
                        pstats['oob'].append(epinfo[d]['out_of_bounds'])
                        pstats['s_c'].append(epinfo[d]['start_coverage'])
                        pstats['s_iv'].append(epinfo[d]['start_variance_inv'])
                        pstats['nsteps'].append(epinfo[d]['num_steps'])
                        episode_rews[d] = 0
                nb_epis = len(pstats['r'])
                logger.info('\nStats only for {} completed episodes:'.format(nb_epis))
                logger.info('Play rewards: {:.3f} +/- {:.1f}'.format(
                        np.mean(pstats['r']), np.std(pstats['r'])))
                logger.info('rewards max/min/median: {:.3f}, {:.3f}, {:.3f}'.format(
                        np.max(pstats['r']), np.min(pstats['r']), np.median(pstats['r'])))
                logger.info('Num steps : {:.3f} +/- {:.1f}'.format(
                        np.mean(pstats['nsteps']), np.std(pstats['nsteps'])))
                logger.info('Start inv-var:  {:.3f} +/- {:.1f}'.format(
                        np.mean(pstats['s_iv']), np.std(pstats['s_iv'])))
                logger.info('Final inv-var:  {:.3f} +/- {:.1f}'.format(
                        np.mean(pstats['iv']), np.std(pstats['iv'])))
                logger.info('Start coverage: {:.3f} +/- {:.1f}'.format(
                        np.mean(pstats['s_c']), np.std(pstats['s_c'])))
                logger.info('Final coverage: {:.3f} +/- {:.1f}'.format(
                        np.mean(pstats['c']), np.std(pstats['c'])))
                logger.info('Final coverage max/min/median: {:.3f}, {:.3f}, {:.3f}'.format(
                        np.max(pstats['c']), np.min(pstats['c']), np.median(pstats['c'])))
                logger.info('Out of bounds total: {}'.format(np.sum(pstats['oob'])))
                logger.info('')
                nb_success = np.sum( np.array(pstats['r']) > success_threshold )
                logger.info('Num exceeding coverage thresh: {} / {}'.format(nb_success, nb_epis))
                logger.info('')
                if len(pstats['r']) >= NB_EVAL_EPIS:
                    logger.info('\nDONE w/{} epis, breaking ...\n'.format(NB_EVAL_EPIS))
                    date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
                    dname = 'policy-rollout-{}-{}-epis-{}-seed-{}-depth-{}-forcegrab-{}-stats-{}.pkl'.format(
                                args.alg,
                                NB_EVAL_EPIS,
                                cloth_config['init']['type'],
                                cloth_config['seed'],
                                cloth_config['env']['use_depth'],
                                cloth_config['env']['force_grab'],
                                date)
                    dname = osp.join('logs', dname)
                    with open(dname, 'wb') as fh:
                        pickle.dump(pstats, fh)
                    logger.info('saving at: {}'.format(dname))
                    break
                # Daniel: env.step() above internally calls reset when we have DummyVecEnv
                # Tbh, I am still not sure why baselines includes an extra env.reset() here?
                if args.env != 'Cloth-v0':
                    obs = env.reset()
    env.close()
    return model


if __name__ == '__main__':
    main(sys.argv)
