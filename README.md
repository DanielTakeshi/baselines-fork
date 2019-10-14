# AUTOLAB Stuff

To install:

- Clone and install `gym_cloth`, from [our cloth repository][1].
- Use `pip install tensorflow-gpu==1.13.1`, though I'm sure other versions work.
- Clone and install this repository via `pip install -e .`
- Install Blender 2.79 or 2.80. (EDIT: we only use 2.79 for this work.)

Let's see if this can give us some interesting results.  Try running something
like this:

```
python -m baselines.run --alg=ppo2 --env=Cloth-v0 --num_env=1 --num_timesteps=1e5 \
        --cloth_config=../gym-cloth/cfg/demo_baselines.yaml \
        --init_state=../gym-cloth/examples/state_init_81_coverage.pkl
```

or for DDPG with a smaller `rb_size` to avoid too much memory allocation, and
with demonstrations, and with a fixed `--init_state` which should be avoided in
general:

```
python -m baselines.run --alg=ddpg --env=Cloth-v0 --num_env=1 --num_timesteps=1e5 \
        --cloth_config=../gym-cloth/cfg/demo_baselines.yaml  --rb_size=1000 \
        --init_state=../gym-cloth/examples/state_init_81_coverage.pkl \
        --demos_path=../gym-cloth/logs/demos-2019-07-02-11-06-clip_a-True-delta_a-True-obs-blender.pkl
```

or with DAgger:

```
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=1 --num_timesteps=1e5 \
        --cloth_config=../gym-cloth/cfg/demo_baselines.yaml --rb_size=25000 \
        --demos_path=../gym-cloth/logs/demos-2019-08-02-15-48-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier1_epis_1000.pkl
```

To do pure behavior cloning, just set `--num_timesteps=1`.

The output goes to a log directory. **Do not open the `0.x.monitor.csv` files
which we use to store results for each episode during training** --- I think
this causes issue with certain numbers in the files not making sense. Do a `cat
0.x.monitor.csv` if you really need to inspect the output during training. Same
for the `log.txt` file.

Command line arguments:

- Current algorithms supported are `ppo2` and `ddpg`. If one pair of command
  line arguments works for one algorithm, it should work for the other(s).

- We pass the environment configuration `--cloth_config` as input. **In this
  configuration file, watch out for**:

  - whether rendering is on *(usually should be False)*.
  - whether we clip action space or not, but "clipping" is misleading since it
    also means the [0,1] action ranges get *expanded* into the [-1,1] range
    *(usually should be True)*.
  - whether we are using delta actions or not, alternative is length/angle
    *(usually should be True)*.
  - the representation of the cloth ('1d' or 'blender'), *usually should be
    blender*.

  The cloth representation in the configuration will automatically determine
  the network type of `mlp` (for 1d) or `cloth_cnn` (for blender/images). It
  should apply for DAgger, DDPG and PPO.

- `--num_env` is the number of environments to run in parallel. For more than
  one environment, you need rendering *off*. Use the `--render_path` argument
  to point to where the renderer is compiled. See the visualize command below
  for an example (use the same argument).

- (Optional) If you do not supply `--seed`, the code will search for a `seed`
  in the `--cloth_config` file. A `--seed` on the command line has priority.

- (Optional) `--init_state` makes the environment start at a specific state.
  It's ideal to avoid this because having initial state diversity matters. But
  it's useful for debugging and rapid experiments.

- (Optional) `--save_path` specifies a directory to save.  Otherwise, it goes
  to `/tmp/openai-[date]`. That's already convenient so I don't normally feel
  the need to specify this argument.

- (Optional) `--demos_path` for demonstrations from `gym-cloth` scripts.

To define other hyperparameters, look at the `cloth()` method inside the
`defaults.py` file, mirroring the `mujoco()` and `atari()` methods. These are
currently supported for PPO2 and DDPG to reduce the number of arguments we have
to supply manually.

- In PPO and DDPG we can set the `num_hidden` which controls the size of our
  network, which by default is fully connected with architecture
  `(input)-h1-h2-output` and tanh activations, but for now `h1=h2=num_hidden`
  so we just input one integer in `{ddpg,ppo2}/defaults.py`. In PPO, if we want
  to clip the action space in [-1,1] (this also means *expanding* those that
  are just in [0,1], somewhat confusingly) this means [inserting another tanh
  in a hacky way][2].  Fortunately, DDPG handles this for us; [the
  `ddpg/models.py` code][3] shows that the actor gets a tanh at the end.
  **However, this also means that DDPG should not be tested without clipping
  action space, right?**

- In PPO and DDPG, `save_interval=1` is used to save the model, and we save
  frequently just in case.

- DDPG also has `normalize_observations` and `normalize_returns` settings. For
  images, I would set the observations to not normalize since we already scale
  by 255. For PPO I don't think these settings exist, for whatever reason, but
  our network design will scale by 255 just like in DDPG.

After training, load and visualize using examples like these:

```
python -m baselines.run --alg=ppo2 --env=Cloth-v0 --num_env=1 --num_timesteps=0 \
        --cloth_config=../gym-cloth/cfg/demo_baselines.yaml \
        --load_path=logs/openai-2019-04-29-20-48-56-212918/checkpoints/00001 \
        --render_path=../gym-cloth/render/build/ --play
```

or

```
python -m baselines.run --alg=ddpg --env=Cloth-v0 --num_env=1 --num_timesteps=0 \
        --cloth_config=../gym-cloth/cfg/demo_baselines.yaml  --rb_size=20000 \
        --load_path=logs/openai-2019-07-04-11-38-46-424501/checkpoints/00000 \
        --render_path=../gym-cloth/render/build/  --play
```

where the `--load_path` should specify the checkpoint from training. Here, I've
used checkpoint 00001, from early on in training. Reminders:

- Set `num_timesteps=0` if we are *playing* the policy. The `--play` just means
  we ignore training and let it run (but we do need to load a model). If we
  want to load a model but do not use `--play`, then that is like pre-training
  another policy. We can do that but I don't anticipate us using this feature.

- **You will need to set the `demo_baselines.yaml` file to enable the
  visualization**, by default it's off as I'm not sure how to get multiple
  visualizations in parallel, and of course, to ensure that the render path
  points to a *compiled* render from the gym-cloth repository.

- **I'm not sure if loading and resuming training will work yet** and we should
  test that if possible, particularly for DDPG and for anything that involves
  normalization (e.g., DDPG normalizing with TensorFlow operations).

- **The arguments need to match with what we originally trained**. If we
  trained using a network with 200 hidden units, but we specify 64 on the
  command line, we (fortunately) get an error message specifying that.

Plot by running `python scripts/{ddpg,ppo2,imit}_quick.py`.

**Blender**: download the `tar.bz2` file corresponding to version 2.80. Then,
untar it with a command like:

```
tar -xvjf blender-2.79b-linux-glibc219-x86_64.tar.bz2
```

and something like this to your `~/.bashrc`:

```
export PATH=${HOME}/blender-2.79b-linux-glibc219-x86_64:$PATH
```

so that you can call `blender` from the command line. **However, it's not
currently clear how to do this on headless servers, even with `--background`
enabled, so we are currently using Blender 2.79**. We have also tested this on
Mac OS X systems.


[1]:https://github.com/BerkeleyAutomation/gym-cloth
[2]:https://github.com/BerkeleyAutomation/baselines-fork/blob/master/baselines/common/policies.py
[3]:https://github.com/BerkeleyAutomation/baselines-fork/blob/master/baselines/ddpg/models.py


# OpenAI Documentation Below (leave for clarity)

**Status:** Active (under active development, breaking changes may occur)

<img src="data/logo.jpg" width=25% align="right" /> [![Build status](https://travis-ci.org/openai/baselines.svg?branch=master)](https://travis-ci.org/openai/baselines)

# Baselines

OpenAI Baselines is a set of high-quality implementations of reinforcement learning algorithms.

These algorithms will make it easier for the research community to replicate, refine, and identify new ideas, and will create good baselines to build research on top of. Our DQN implementation and its variants are roughly on par with the scores in published papers. We expect they will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones.

## Prerequisites
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows
### Ubuntu

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```

## Virtual environment
From the general python package sanity perspective, it is a good idea to use virtual environments (virtualenvs) to make sure packages from different projects do not interfere with each other. You can install virtualenv (which is itself a pip package) via
```bash
pip install virtualenv
```
Virtualenvs are essentially folders that have copies of python executable and all python packages.
To create a virtualenv called venv with python3, one runs
```bash
virtualenv /path/to/venv --python=python3
```
To activate a virtualenv:
```
. /path/to/venv/bin/activate
```
More thorough tutorial on virtualenvs and options can be found [here](https://virtualenv.pypa.io/en/stable/)


## Installation
- Clone the repo and cd into it:
    ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines
    ```
- If you don't have TensorFlow installed already, install your favourite flavor of TensorFlow. In most cases,
    ```bash
    pip install tensorflow-gpu # if you have a CUDA-compatible gpu and proper drivers
    ```
    or
    ```bash
    pip install tensorflow
    ```
    should be sufficient. Refer to [TensorFlow installation guide](https://www.tensorflow.org/install/)
    for more details.

- Install baselines package
    ```bash
    pip install -e .
    ```

### MuJoCo
Some of the baselines examples use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py)

## Testing the installation
All unit tests in baselines can be run using pytest runner:
```
pip install pytest
pytest
```

## Training models
Most of the algorithms in baselines repo are used as follows:
```bash
python -m baselines.run --alg=<name of the algorithm> --env=<environment_id> [additional arguments]
```
### Example 1. PPO with MuJoCo Humanoid
For instance, to train a fully-connected network controlling MuJoCo humanoid using PPO2 for 20M timesteps
```bash
python -m baselines.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num_timesteps=2e7
```
Note that for mujoco environments fully-connected network is default, so we can omit `--network=mlp`
The hyperparameters for both network and the learning algorithm can be controlled via the command line, for instance:
```bash
python -m baselines.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num_timesteps=2e7 --ent_coef=0.1 --num_hidden=32 --num_layers=3 --value_network=copy
```
will set entropy coefficient to 0.1, and construct fully connected network with 3 layers with 32 hidden units in each, and create a separate network for value function estimation (so that its parameters are not shared with the policy network, but the structure is the same)

See docstrings in [common/models.py](baselines/common/models.py) for description of network parameters for each type of model, and
docstring for [baselines/ppo2/ppo2.py/learn()](baselines/ppo2/ppo2.py#L152) for the description of the ppo2 hyperparameters.

### Example 2. DQN on Atari
DQN with Atari is at this point a classics of benchmarks. To run the baselines implementation of DQN on Atari Pong:
```
python -m baselines.run --alg=deepq --env=PongNoFrameskip-v4 --num_timesteps=1e6
```

## Saving, loading and visualizing models
The algorithms serialization API is not properly unified yet; however, there is a simple method to save / restore trained models.
`--save_path` and `--load_path` command-line option loads the tensorflow state from a given path before training, and saves it after the training, respectively.
Let's imagine you'd like to train ppo2 on Atari Pong,  save the model and then later visualize what has it learnt.
```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=2e7 --save_path=~/models/pong_20M_ppo2
```
This should get to the mean reward per episode about 20. To load and visualize the model, we'll do the following - load the model, train it for 0 steps, and then visualize:
```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=0 --load_path=~/models/pong_20M_ppo2 --play
```

*NOTE:* At the moment Mujoco training uses VecNormalize wrapper for the environment which is not being saved correctly; so loading the models trained on Mujoco will not work well if the environment is recreated. If necessary, you can work around that by replacing RunningMeanStd by TfRunningMeanStd in [baselines/common/vec_env/vec_normalize.py](baselines/common/vec_env/vec_normalize.py#L12). This way, mean and std of environment normalizing wrapper will be saved in tensorflow variables and included in the model file; however, training is slower that way - hence not including it by default

## Loading and vizualizing learning curves and other training metrics
See [here](docs/viz/viz.ipynb) for instructions on how to load and display the training data.

## Subpackages

- [A2C](baselines/a2c)
- [ACER](baselines/acer)
- [ACKTR](baselines/acktr)
- [DDPG](baselines/ddpg)
- [DQN](baselines/deepq)
- [GAIL](baselines/gail)
- [HER](baselines/her)
- [PPO1](baselines/ppo1) (obsolete version, left here temporarily)
- [PPO2](baselines/ppo2)
- [TRPO](baselines/trpo_mpi)



## Benchmarks
Results of benchmarks on Mujoco (1M timesteps) and Atari (10M timesteps) are available
[here for Mujoco](https://htmlpreview.github.com/?https://github.com/openai/baselines/blob/master/benchmarks_mujoco1M.htm)
and
[here for Atari](https://htmlpreview.github.com/?https://github.com/openai/baselines/blob/master/benchmarks_atari10M.htm)
respectively. Note that these results may be not on the latest version of the code, particular commit hash with which results were obtained is specified on the benchmarks page.

To cite this repository in publications:

    @misc{baselines,
      author = {Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai and Zhokhov, Peter},
      title = {OpenAI Baselines},
      year = {2017},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/openai/baselines}},
    }
