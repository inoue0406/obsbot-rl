#!/bin/bash

case="result_20230528_obsbot-rl_pointtest"

# Running RL-based obsbot script
python ../src/main_obsbot_rl.py --env_name ObsBot2DPoint\
       --result_path $case\
       --start_timesteps 10000\
       --eval_freq 5000\
       --max_timesteps 2000000\
       --save_models \
       --save_env_vid \
       --expl_noise 0.2\
       --batch_size 100\
       --discount 0.99\
       --tau 0.005\
       --policy_noise 0.1\
       --noise_clip 0.5\
       --policy_freq 2\
       
