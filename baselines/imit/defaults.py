
# Daniel: I did this empirically.
# eval steps is 1k to get a full episode in for the envs like Ant-v2, in which
# all episodes last 1k steps. But we might want to make this one smaller.

def mujoco():
    return dict(
        nb_epoch_cycles=20,
        nb_rollout_steps=100,
        nb_train_steps=50,
        nb_eval_steps=1000,
        noise_type='normal_0.1',
        rb_size=1e6,
    )


# Daniel: for cloth
def cloth():
    return dict(
        nb_epoch_cycles=1,      # I'd like to see log results more frequently
        nb_rollout_steps=10,    # `nsteps` in PPO2, though I'm doing 10 so we update more often
        nb_train_steps=200,     # High b/c Deepmind did '40 learning updates per env step'
        nb_eval_steps=20,
        batch_size=128,
        num_hidden=200,         # default architecture, similar to PPO2
        rb_size=5e4,            # Sim2Real @CoRL used 5e4. Careful about RAM!
        save_interval=1,
        bc_epochs=500,          # I added.
        actor_l2_reg=1e-5,      # We really need regularization for IL.
    )
