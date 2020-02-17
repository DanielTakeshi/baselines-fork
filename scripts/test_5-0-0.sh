# LOOK AT 5.0.0 EXPERIMENTS. THESE ARE RGBD. Use 249 as the checkpoint, since that corresponds to 50k steps.
# Make sure the config file is correct and has seed 1600. Once again BE CAREFUL ABOUT THE CFG FILES!
# The name is `gym-cloth-iros2020` but the actual repo is here: https://github.com/DanielTakeshi/gym-cloth

CFG1=../gym-cloth-iros2020/cfg/t1_rgbd.yaml
CFG2=../gym-cloth-iros2020/cfg/t2_rgbd.yaml
CFG3=../gym-cloth-iros2020/cfg/t3_rgbd.yaml
POL1=../policies-cloth-sim-iros2020/openai-2020-02-11-15-52-56-391653/checkpoints
POL2=../policies-cloth-sim-iros2020/openai-2020-02-12-14-11-03-092908/checkpoints
POL3=../policies-cloth-sim-iros2020/openai-2020-02-15-19-37-59-883528/checkpoints
NENV=10

# TIER 1:
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG1} --rb_size=10000 --load_path=${POL1}/bc_epoch_0010 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG1} --rb_size=10000 --load_path=${POL1}/bc_epoch_0050 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG1} --rb_size=10000 --load_path=${POL1}/bc_epoch_0100 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG1} --rb_size=10000 --load_path=${POL1}/bc_epoch_0200 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG1} --rb_size=10000 --load_path=${POL1}/bc_epoch_0300 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG1} --rb_size=10000 --load_path=${POL1}/bc_epoch_0400 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG1} --rb_size=10000 --load_path=${POL1}/bc_epoch_0500 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG1} --rb_size=10000 --load_path=${POL1}/00040         --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG1} --rb_size=10000 --load_path=${POL1}/00080         --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG1} --rb_size=10000 --load_path=${POL1}/00120         --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG1} --rb_size=10000 --load_path=${POL1}/00160         --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG1} --rb_size=10000 --load_path=${POL1}/00200         --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG1} --rb_size=10000 --load_path=${POL1}/00249         --play

# TIER 2:
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG2} --rb_size=10000 --load_path=${POL2}/bc_epoch_0010 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG2} --rb_size=10000 --load_path=${POL2}/bc_epoch_0050 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG2} --rb_size=10000 --load_path=${POL2}/bc_epoch_0100 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG2} --rb_size=10000 --load_path=${POL2}/bc_epoch_0200 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG2} --rb_size=10000 --load_path=${POL2}/bc_epoch_0300 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG2} --rb_size=10000 --load_path=${POL2}/bc_epoch_0400 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG2} --rb_size=10000 --load_path=${POL2}/bc_epoch_0500 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG2} --rb_size=10000 --load_path=${POL2}/00040         --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG2} --rb_size=10000 --load_path=${POL2}/00080         --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG2} --rb_size=10000 --load_path=${POL2}/00120         --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG2} --rb_size=10000 --load_path=${POL2}/00160         --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG2} --rb_size=10000 --load_path=${POL2}/00200         --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG2} --rb_size=10000 --load_path=${POL2}/00249         --play

# TIER 3:
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG3} --rb_size=10000 --load_path=${POL3}/bc_epoch_0010 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG3} --rb_size=10000 --load_path=${POL3}/bc_epoch_0050 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG3} --rb_size=10000 --load_path=${POL3}/bc_epoch_0100 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG3} --rb_size=10000 --load_path=${POL3}/bc_epoch_0200 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG3} --rb_size=10000 --load_path=${POL3}/bc_epoch_0300 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG3} --rb_size=10000 --load_path=${POL3}/bc_epoch_0400 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG3} --rb_size=10000 --load_path=${POL3}/bc_epoch_0500 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG3} --rb_size=10000 --load_path=${POL3}/00040         --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG3} --rb_size=10000 --load_path=${POL3}/00080         --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG3} --rb_size=10000 --load_path=${POL3}/00120         --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG3} --rb_size=10000 --load_path=${POL3}/00160         --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG3} --rb_size=10000 --load_path=${POL3}/00200         --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=${NENV} --num_timesteps=0 --cloth_config=${CFG3} --rb_size=10000 --load_path=${POL3}/00249         --play
