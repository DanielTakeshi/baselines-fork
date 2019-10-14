# The DEPTH based stuff!!
# BUT MAKE SURE THE 'FIXED' BASELINES USE SEED 1600 ... yeah I know it's a pain.
# ALSO, force_grab should be true in the config.
# Otherwise use the fixed_t1, fixed_t2, fixed_t3 stuff as usual ...

# TIER 1 (openai-2019-09-02-23-01-55-146225) ...

#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-01-55-146225/checkpoints/bc_epoch_0010  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-01-55-146225/checkpoints/bc_epoch_0050  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-01-55-146225/checkpoints/bc_epoch_0100  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-01-55-146225/checkpoints/bc_epoch_0200  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-01-55-146225/checkpoints/bc_epoch_0300  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-01-55-146225/checkpoints/bc_epoch_0400  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-01-55-146225/checkpoints/bc_epoch_0500  --play
#
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-01-55-146225/checkpoints/00040 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-01-55-146225/checkpoints/00080 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-01-55-146225/checkpoints/00120 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-01-55-146225/checkpoints/00160 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-01-55-146225/checkpoints/00200 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t1.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-23-01-55-146225/checkpoints/00249 --play



# Do Tier 2 (openai-2019-09-04-12-56-54-072948) ...

python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-04-12-56-54-072948/checkpoints/bc_epoch_0010  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-04-12-56-54-072948/checkpoints/bc_epoch_0050  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-04-12-56-54-072948/checkpoints/bc_epoch_0100  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-04-12-56-54-072948/checkpoints/bc_epoch_0200  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-04-12-56-54-072948/checkpoints/bc_epoch_0300  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-04-12-56-54-072948/checkpoints/bc_epoch_0400  --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-04-12-56-54-072948/checkpoints/bc_epoch_0500  --play

python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-04-12-56-54-072948/checkpoints/00040 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-04-12-56-54-072948/checkpoints/00080 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-04-12-56-54-072948/checkpoints/00120 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-04-12-56-54-072948/checkpoints/00160 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-04-12-56-54-072948/checkpoints/00200 --play
python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t2.yaml --rb_size=10000 \
       --load_path=../policies-cloth-sim/openai-2019-09-04-12-56-54-072948/checkpoints/00249 --play



# Do Tier 3 (openai-2019-09-02-19-25-40-802588) ...

#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-19-25-40-802588/checkpoints/bc_epoch_0010  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-19-25-40-802588/checkpoints/bc_epoch_0050  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-19-25-40-802588/checkpoints/bc_epoch_0100  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-19-25-40-802588/checkpoints/bc_epoch_0200  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-19-25-40-802588/checkpoints/bc_epoch_0300  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-19-25-40-802588/checkpoints/bc_epoch_0400  --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-19-25-40-802588/checkpoints/bc_epoch_0500  --play
#
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-19-25-40-802588/checkpoints/00040 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-19-25-40-802588/checkpoints/00080 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-19-25-40-802588/checkpoints/00120 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-19-25-40-802588/checkpoints/00160 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-19-25-40-802588/checkpoints/00200 --play
#python -m baselines.run --alg=imit --env=Cloth-v0 --num_env=8 --num_timesteps=0 --cloth_config=../gym-cloth/cfg/demo_baselines_fixed_t3.yaml --rb_size=10000 \
#       --load_path=../policies-cloth-sim/openai-2019-09-02-19-25-40-802588/checkpoints/00249 --play
