import os
import numpy as np
import matplotlib.pyplot as plt
import time

from env_obsbot_2dmov import ObsBot2D
from model_TD3 import *
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

    # Create a folder for saving models
    if not os.path.exists("./%s" % opt.result_path):
        os.makedirs("./%s" % opt.result_path)
    if opt.save_models and not os.path.exists(model_path):
        os.makedirs(model_path)

    # Create an environment
    num_bots = 10
    env = ObsBot2D(num_bots)

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

    # Create new folder to store final results
    def mkdir(base, name):
        path = os.path.join(base, name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    work_dir = mkdir("exp","brs")
    monitor_dir = mkdir(work_dir, "monitor")
    max_episode_steps = env._max_episode_steps
    save_env_vid = False
    if save_env_vid:
        env = wrappers.Monitor(env, monitor_dir, force=True)
        env.reset()    

    # Initialize the loop
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    t0 = time.time()

    # Training Loop

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
                evaluations.append(evaluate_policy(policy,env))
                policy.save(file_name, directory=model_path)
                np.save("./%s/%s" % (opt.result_path,file_name), evaluations)
    
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
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
  
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

#    # create result dir
#    if not os.path.exists(opt.result_path):
#        os.mkdir(opt.result_path)
#    
#    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
#        opt_file.write(json.dumps(vars(opt), indent=2))
#
##    # Tracking by MLFlow
##    experiment_id = mlflow.tracking.MlflowClient().get_experiment_by_name(opt.result_path[0:10])
#    
#    # generic log file  
#    logfile = open(os.path.join(opt.result_path, 'log_run.txt'),'w')
#    logfile.write('Start time:'+time.ctime()+'\n')
#    tstart = time.time()
#
#    # model information
#    modelinfo = open(os.path.join(opt.result_path, 'model_info.txt'),'w')
#            
#    modelinfo.write('Model Structure \n')
#    modelinfo.write(str(model))
#    count_parameters(model,modelinfo)
#    modelinfo.close()
#        
#    # Prep logger
#    train_logger = Logger(
#        os.path.join(opt.result_path, 'train.log'),
#        ['epoch', 'loss', 'lr'])
#    
#    # training 
#
#
#    # output elapsed time
#    logfile.write('End time: '+time.ctime()+'\n')
#    tend = time.time()
#    tdiff = float(tend-tstart)/3600.0
#    logfile.write('Elapsed time[hours]: %f \n' % tdiff)
