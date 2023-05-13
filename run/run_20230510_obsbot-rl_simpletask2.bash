#!/bin/bash

case="result_20230510_obsbot-rl_simpletask2"

# Running RL-based obsbot script
python ../src/main_obsbot_rl.py --env_name ObsBot2D\
       --result_path $case\
       --start_timesteps 10000\
       --eval_freq 5000\
       --max_timesteps 15000\
       --save_models\
       --expl_noise 0.1\
       --batch_size 100\
       --discount 0.99\
       --tau 0.005\
       --policy_noise 0.2\
       --noise_clip 0.5\
       --policy_freq 2\
       
