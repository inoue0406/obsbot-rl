import os
import numpy as np
import time
import json

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from utils import Logger

import obsbot_env
from model_TD3 import ReplayBuffer, TD3, evaluate_policy
from opts import parse_opts

def count_parameters(model,f):
    for name,p in model.named_parameters():
        f.write("name,"+name+", Trainable, "+str(p.requires_grad)+",#params, "+str(p.numel())+"\n")
    Nparam = sum(p.numel() for p in model.parameters())
    Ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f.write("Number of params:"+str(Nparam)+", Trainable parameters:"+str(Ntrain)+"\n")
    
if __name__ == '__main__':
   
    # parse command-line options
    opt = parse_opts()
    print(json.dumps(vars(opt), indent=2))

    seed = 0 # Random seed number
    
    file_name = "%s_%s_%s" % ("TD3", opt.env_name, str(seed))
    print ("---------------------------------------")
    print ("Settings: %s" % (file_name))
    print ("---------------------------------------")

    model_path = os.path.join(".",opt.result_path, "models")
    monitor_dir = os.path.join(".",opt.result_path, "monitor")

    # Create a folder for saving models
    if not os.path.exists("./%s" % opt.result_path):
        os.makedirs("./%s" % opt.result_path)
    if opt.save_models and not os.path.exists(model_path):
        os.makedirs(model_path)

    # Create an environment
    env = gym.make(opt.env_name,
                   num_bots=opt.num_bots,
                   max_episode_steps=opt.max_episode_steps,
                   metfield_path=opt.metfield_path,
                   init_mode=opt.init_mode,
                   action_scale=opt.action_scale)

    # Set seeds etc.
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0]) # max value of action variable

    # Create a policy network (TD3 model)
    policy = TD3(state_dim, action_dim, max_action)

    # Create replay memory
    replay_buffer = ReplayBuffer()

    # Define a list to store eval results
    evaluations = [evaluate_policy(policy,env)]

    max_episode_steps = env._max_episode_steps

    if opt.save_env_vid:
        env = wrappers.Monitor(env, monitor_dir, force = True)
    
    # Initialize the loop
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True

    # generic log file
    logfile = open(os.path.join(opt.result_path, 'log_run.txt'),'w')
    logfile.write('Start time:'+time.ctime()+'\n')
    tstart = time.time()

    # Prep a logger for recording reward
    train_logger = Logger(
        os.path.join(opt.result_path, 'train.log'),
        ['episode', 'reward','replay_buffer_size'])

    while total_timesteps < opt.max_timesteps:
        # If the episode is done
        if done:

            # If we are not at the very beginning, we start the training process of the model
            if total_timesteps != 0:
                print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
                policy.train(replay_buffer, episode_timesteps,
                             opt.batch_size, opt.discount,
                             opt.tau, opt.policy_noise,
                             opt.noise_clip, opt.policy_freq)

            # We evaluate the episode and we save the policy
            if timesteps_since_eval >= opt.eval_freq:
                timesteps_since_eval %= opt.eval_freq
                reward = evaluate_policy(policy,env)
                evaluations.append(reward)
                policy.save(file_name, directory=model_path)
                np.save("./%s/%s" % (opt.result_path,file_name), evaluations)
                train_logger.log({
                    'episode': episode_num,
                    'reward': reward,
                    'replay_buffer_size': len(replay_buffer.storage)})
    
            # When the training step is done, we reset the state of the environment
            obs = env.reset()
    
            # Set the Done to False
            done = False
    
            # Set rewards and episode timesteps to zero
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
  
        # Before 10000 timesteps, we play random actions
        if total_timesteps < opt.start_timesteps:
            action = env.action_space.sample()
        else: # After 10000 timesteps, we switch to the model
            action = policy.select_action(np.array(obs))
            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            if opt.expl_noise != 0:
                action = (action + np.random.normal(0, opt.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)
  
        # The agent performs the action in the environment, then reaches the next state and receives the reward
        new_obs, reward, done, _ = env.step(action)
  
        # We check if the episode is done
        done_bool = 0 if episode_timesteps + 1 == max_episode_steps else float(done)
  
        # We increase the total reward
        episode_reward += reward
  
        # We store the new transition into the Experience Replay memory (ReplayBuffer)
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
        obs = new_obs
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    # We add the last policy evaluation to our list of evaluations and we save our model
    evaluations.append(evaluate_policy(policy,env))
    if opt.save_models: policy.save("%s" % (file_name), directory=model_path)
    np.save("./%s/%s" % (opt.result_path,file_name), evaluations)

    # output elapsed time
    logfile.write('End time: '+time.ctime()+'\n')
    tend = time.time()
    tdiff = float(tend-tstart)/3600.0
    logfile.write('Elapsed time[hours]: %f \n' % tdiff)
    
    # Save learned action as a video
    eval_episodes = 10
    if not opt.save_env_vid:
        env = wrappers.Monitor(env, monitor_dir, force = True)
        
    env.reset()

    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    policy = TD3(state_dim, action_dim, max_action)
    policy.load(file_name, model_path)
    _ = evaluate_policy(policy, env, eval_episodes=eval_episodes)
