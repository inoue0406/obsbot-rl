#!/bin/bash

case="result_20230710_obsbot-rl_n50_scl5_noise01"

# Running RL-based obsbot script
python ../src/main_obsbot_rl.py --env_name obsbot2dpoint-v0\
       --result_path $case\
       --metfield_path ../data/artfield/vzero_256/\
       --num_bots 50\
       --max_episode_steps 50\
       --action_scale 0.5\
       --start_timesteps 10000\
       --eval_freq 5000\
       --max_timesteps 2000000\
       --save_models \
       --save_env_vid \
       --expl_noise 0.1\
       --batch_size 100\
       --discount 0.99\
       --tau 0.005\
       --policy_noise 0.1\
       --noise_clip 0.5\
       --policy_freq 2\
       
