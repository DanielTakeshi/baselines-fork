# LOOK AT 4.5.0 EXPERIMENTS.
# The color analogue for 6.2.0 experiments. NOTE 6.1.0 !! These have 200 and 249 as checkpoints.
# BUT MAKE SURE THE 'FIXED' BASELINES USE SEED 1600 ... yeah I know it's a pain.
# ALSO, force_grab should be true in the config.
# Otherwise use the fixed_t1_color, fixed_t2_color, fixed_t3_color stuff as usual ...

# TIER 1 (openai-2019-09-02-23-03-41-443793) ...

#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-03-41-443793/checkpoints/bc_epoch_0010  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-03-41-443793/checkpoints/bc_epoch_0050  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-03-41-443793/checkpoints/bc_epoch_0100  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-03-41-443793/checkpoints/bc_epoch_0200  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-03-41-443793/checkpoints/bc_epoch_0300  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-03-41-443793/checkpoints/bc_epoch_0400  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-03-41-443793/checkpoints/bc_epoch_0500  --play
#
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-03-41-443793/checkpoints/00040 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-03-41-443793/checkpoints/00080 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-03-41-443793/checkpoints/00120 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-03-41-443793/checkpoints/00160 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-03-41-443793/checkpoints/00200 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-03-41-443793/checkpoints/00249 --play


# Do Tier 2 (openai-2019-09-01-20-37-45-609860) ...

python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-20-37-45-609860/checkpoints/bc_epoch_0010  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-20-37-45-609860/checkpoints/bc_epoch_0050  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-20-37-45-609860/checkpoints/bc_epoch_0100  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-20-37-45-609860/checkpoints/bc_epoch_0200  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-20-37-45-609860/checkpoints/bc_epoch_0300  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-20-37-45-609860/checkpoints/bc_epoch_0400  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-20-37-45-609860/checkpoints/bc_epoch_0500  --play

python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-20-37-45-609860/checkpoints/00040 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-20-37-45-609860/checkpoints/00080 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-20-37-45-609860/checkpoints/00120 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-20-37-45-609860/checkpoints/00160 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-20-37-45-609860/checkpoints/00200 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-20-37-45-609860/checkpoints/00249 --play



# Do Tier 3 (openai-2019-09-02-19-30-13-323241) ...

python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-02-19-30-13-323241/checkpoints/bc_epoch_0010  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-02-19-30-13-323241/checkpoints/bc_epoch_0050  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-02-19-30-13-323241/checkpoints/bc_epoch_0100  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-02-19-30-13-323241/checkpoints/bc_epoch_0200  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-02-19-30-13-323241/checkpoints/bc_epoch_0300  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-02-19-30-13-323241/checkpoints/bc_epoch_0400  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-02-19-30-13-323241/checkpoints/bc_epoch_0500  --play

python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-02-19-30-13-323241/checkpoints/00040 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-02-19-30-13-323241/checkpoints/00080 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-02-19-30-13-323241/checkpoints/00120 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-02-19-30-13-323241/checkpoints/00160 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-02-19-30-13-323241/checkpoints/00200 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-02-19-30-13-323241/checkpoints/00249 --play
