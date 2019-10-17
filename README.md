# AUTOLAB Stuff

To install:

- Clone and install `gym_cloth`, from [our cloth repository][1].
- Use `pip install tensorflow-gpu==1.13.1`, though I'm sure other versions work.
- Clone and install this repository via `pip install -e .`
- Install Blender 2.79 or 2.80. (EDIT: we only use 2.79 for this work.)

Let's see if this can give us some interesting results.  Try running something
like this for DAgger:

```
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=1 --num_timesteps=1e5 \
        --cloth_config=../gym-cloth/cfg/demo_baselines.yaml --rb_size=25000 \
        --demos_path=../gym-cloth/logs/demos-2019-08-02-15-48-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier1_epis_1000.pkl
```

To do pure behavior cloning, just set `--num_timesteps=1`. PPO and DDPG are also
supported.

The output goes to a log directory. **Do not open the `0.x.monitor.csv` files
which we use to store results for each episode during training** --- I think
this causes issue with certain numbers in the files not making sense. Do a `cat
0.x.monitor.csv` if you really need to inspect the output during training. Same
for the `log.txt` file.

Command line arguments:

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
