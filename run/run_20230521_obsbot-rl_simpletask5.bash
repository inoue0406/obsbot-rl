#!/bin/bash

case="result_20230521_obsbot-rl_simpletask5"

# Running RL-based obsbot script
python ../src/main_obsbot_rl.py --env_name ObsBot2D\
       --result_path $case\
       --start_timesteps 10000\
       --eval_freq 5000\
       --max_timesteps 500000\
       --save_models \
       --expl_noise 0.2\
       --batch_size 100\
       --discount 0.99\
       --tau 0.005\
       --policy_noise 0.1\
       --noise_clip 0.5\
       --policy_freq 2\
       
