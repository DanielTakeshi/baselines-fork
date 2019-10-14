# The color analogue for 6.1.0 experiments.
# BUT MAKE SURE THE 'FIXED' BASELINES USE SEED 1600 ... yeah I know it's a pain.
# ALSO, force_grab should be true in the config.
# Otherwise use the fixed_t1_color, fixed_t2_color, fixed_t3_color stuff as usual ...

# TIER 1 (openai-2019-09-01-11-09-50-855851) ...

#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-01-11-09-50-855851/checkpoints/bc_epoch_0010  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-01-11-09-50-855851/checkpoints/bc_epoch_0050  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-01-11-09-50-855851/checkpoints/bc_epoch_0100  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-01-11-09-50-855851/checkpoints/bc_epoch_0200  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-01-11-09-50-855851/checkpoints/bc_epoch_0300  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-01-11-09-50-855851/checkpoints/bc_epoch_0400  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-01-11-09-50-855851/checkpoints/bc_epoch_0500  --play
#
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-01-11-09-50-855851/checkpoints/00040 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-01-11-09-50-855851/checkpoints/00080 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-01-11-09-50-855851/checkpoints/00120 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-01-11-09-50-855851/checkpoints/00160 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-01-11-09-50-855851/checkpoints/00199 --play


# Do Tier 2 () ...

#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim//checkpoints/bc_epoch_0010  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim//checkpoints/bc_epoch_0050  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim//checkpoints/bc_epoch_0100  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim//checkpoints/bc_epoch_0200  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim//checkpoints/bc_epoch_0300  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim//checkpoints/bc_epoch_0400  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim//checkpoints/bc_epoch_0500  --play

#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim//checkpoints/00040 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim//checkpoints/00080 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim//checkpoints/00120 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim//checkpoints/00160 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2_color.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim//checkpoints/00199 --play


# Do Tier 3 (openai-2019-09-01-11-10-28-087733) ...

python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-11-10-28-087733/checkpoints/bc_epoch_0010  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-11-10-28-087733/checkpoints/bc_epoch_0050  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-11-10-28-087733/checkpoints/bc_epoch_0100  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-11-10-28-087733/checkpoints/bc_epoch_0200  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-11-10-28-087733/checkpoints/bc_epoch_0300  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-11-10-28-087733/checkpoints/bc_epoch_0400  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-11-10-28-087733/checkpoints/bc_epoch_0500  --play

python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-11-10-28-087733/checkpoints/00040 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-11-10-28-087733/checkpoints/00080 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-11-10-28-087733/checkpoints/00120 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-11-10-28-087733/checkpoints/00160 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3_color.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-01-11-10-28-087733/checkpoints/00199 --play
