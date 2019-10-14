def mujoco():
    return dict(
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.0,
        lr=lambda f: 3e-4 * f,
        cliprange=0.2,
        value_network='copy'
    )

# Daniel: more accurately matches the PPO paper.
# nsteps must be horizon T, noptepochs is 3 in the paper but whatever.
# lr annealed with a factor, but clipping parameter doesn't have alpha factor here?
def atari():
    return dict(
        nsteps=128,
        nminibatches=4,
        lam=0.95,
        gamma=0.99,
        noptepochs=4,
        log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=0.1,
    )


# Daniel: for cloth. Using nsteps=20 to match DDPG.
def cloth():
    return dict(
        num_hidden=200,
        save_interval=1,
        nsteps=20,
        nminibatches=4,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=0.1,
    )



def retro():
    return atari()
