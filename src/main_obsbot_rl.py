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

    env_name = "ObsBot2D" # Name of a environment
    seed = 0 # Random seed number
    start_timesteps = 1e4 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
    eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
    max_timesteps = 5e5 # Total number of iterations/timesteps
    save_models = True # Boolean checker whether or not to save the pre-trained model
    expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
    batch_size = 100 # Size of the batch
    discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
    tau = 0.005 # Target network update rate
    policy_noise = 0.2 # STD of Gaussian noise adde d to the actions for the exploration purposes
    noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
    policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated

    file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
    print ("---------------------------------------")
    print ("Settings: %s" % (file_name))
    print ("---------------------------------------")

    # Create a folder for saving models
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

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

    while total_timesteps < max_timesteps:
        # If the episode is done
        if done:

            # If we are not at the very beginning, we start the training process of the model
            if total_timesteps != 0:
                print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
                policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

            # We evaluate the episode and we save the policy
            if timesteps_since_eval >= eval_freq:
                timesteps_since_eval %= eval_freq
                evaluations.append(evaluate_policy(policy,env))
                policy.save(file_name, directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations)
    
            # When the training step is done, we reset the state of the environment
            obs = env.reset()
    
            # Set the Done to False
            done = False
    
            # Set rewards and episode timesteps to zero
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
  
        # Before 10000 timesteps, we play random actions
        if total_timesteps < start_timesteps:
            action = env.action_space.sample()
        else: # After 10000 timesteps, we switch to the model
            action = policy.select_action(np.array(obs))
            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            if expl_noise != 0:
                action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)
  
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
    if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
    np.save("./results/%s" % (file_name), evaluations)

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
