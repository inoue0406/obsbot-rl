"""
ObsBot2DPoint class for simulating a 2D environment with observation bots.
This class inherits from gym.Env and defines methods to simulate the 2d point observation,
such as rainfall or temperature gauges given a 2d meteorological field.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import h5py

import random
import gym

# import interpolator
from obsbot_env.envs.nearest_neighbor_interp_kdtree import nearest_neighbor_interp_kd

class ObsBot2DPoint(gym.Env):
    def __init__(self, num_bots, metfield_path):
        """
        Initialize the environment with the given number of observation bots.
        
        Args:
            num_bots (int): Number of observation bots.
            metfield_path (src): The directory path containing meteorological data (in .h5 format)
        """
        super(ObsBot2DPoint, self).__init__()

        # File path for meteorological field
        self.data_files = glob.glob(metfield_path+"*h5")
        self.data_files = sorted(self.data_files)
        self.data_id = 0
        self.data_id_max = len(self.data_files)
        print("Meteorolgical data path:",metfield_path)
        print("Number of met fields used in the training process:",self.data_id_max)

        # This is to be referenced by Gym Wrappers.
        self.metadata = {'render.modes': ['rgb_array']}
        
        # speed limit for observation bots in [m/s]
        self.speed_limit = 1.0
        
        # define the size of a rectangular arena in [m]
        self.arena_size = 10000.0

        # define XY coordinates of the reward area
        #self.xreward1 = 2000.0
        self.xreward1 = 0.0
        self.xreward2 = 4000.0
        self.yreward1 = -4500.0
        self.yreward2 = 4500.0

        # time step in [s]
        self.dt = 300

        # scaling coefficient for scaling input to the neural network
        self.scaling_coeff = 10000.0
        
        # maximum steps per episode
        self._max_episode_steps = 30
        # run steps for counting up
        self.steps = 0

        self.num_bots = num_bots

        # empty state
        self.state = None

        # Define action space.
        # velocity must be smaller than speed limit.
        # 2 * num_bots for representing vx, vy
        self.action_space = gym.spaces.Box(-self.speed_limit,self.speed_limit, (2*num_bots,), np.float32)

        # Define observation space.
        # (X,Y) position and Meteo field is taken as observation
        self.observation_space = gym.spaces.Box(-self.arena_size/2,self.arena_size/2, (3*self.num_bots,), np.float32)

        # Reward range
        self.reward_range = (0, 1000)

        # Grid size of a meteorological field given as the ground truth
        self.field_height = 256
        self.field_width = 256

    def reset(self):
        """
        Reset the environment to an initial state.
        
        Returns:
            np.array: The initial state of the environment.
        """
        # XY : Start from arbitrary position
        XY_pc = np.random.randint(-self.arena_size/2,self.arena_size/2,2*self.num_bots)
        # Initialize with zero velocity
        self.velocity = np.zeros(2*self.num_bots)

        # Set xy positions of observation bots
        self.x_pc = XY_pc.reshape([2,self.num_bots])[0,:]
        self.y_pc = XY_pc.reshape([2,self.num_bots])[1,:]

        # Set 2d meteorological field
        fpath = self.data_files[self.data_id]
        print("Met field data id and path:",self.data_id," ",fpath)
        h5file = h5py.File(fpath,'r')
        self.R_grd = h5file['R'][()].astype(np.float32)
        # TEMP use initial data only
        self.R_grd = self.R_grd[0,:,:]
        #self.R_grd = np.zeros((self.field_height,self.field_width))
        self.data_id = (self.data_id + 1) % self.data_id_max
        self.R_pc = np.zeros(self.num_bots)

        # Initial state is set as x,y,R.
        self.state=np.concatenate([XY_pc,self.R_pc])

        # Set xy coordinates for 2d grid field
        self.XY_grd = self.xy_grid(self.field_height, self.field_width)
        # scale from [0,1] to  arena size
        self.XY_grd = (self.XY_grd - 0.5) * self.arena_size

        return self.state
    
    def xy_grid(self, height, width):
        # generate constant xy grid in [0,1] range
        x1grd = np.linspace(0, 1, width)  # 1d grid
        y1grd = np.linspace(0, 1, height)  # 1d grid

        Xgrid = np.zeros((height, width))
        Ygrid = np.zeros((height, width))
        for j in range(height):
            Xgrid[j, :] = x1grd
        for k in range(width):
            Ygrid[:, k] = y1grd

        XYgrid = np.stack([Xgrid,Ygrid], axis=0)

        return XYgrid
    
    def grid_to_pc_nearest(self):
        # convert grid to point cloud
        # R_grd: grid value with [batch,channels,height,width] dim
        # XY_pc: point cloud position with [batch,2,N] dim
        #        scaled to [0,1]
        R_grd_exp = np.expand_dims(self.R_grd,axis=[0,1])
        XY_pc_exp = np.expand_dims(np.stack([self.x_pc,self.y_pc], axis=0), axis=0)
        batch,k,height,width = R_grd_exp.shape
        XY_grd_tmp = self.XY_grd.reshape(batch,2,height*width).transpose(0,2,1)
        XY_pc_tmp = XY_pc_exp.transpose(0,2,1)
        R_grd_tmp = R_grd_exp.reshape(batch,k,height*width)
        # interpolate
        R_pc = nearest_neighbor_interp_kd(XY_pc_tmp,XY_grd_tmp,R_grd_tmp)
        return R_pc

    def pc_to_grid_nearest(self):
        # convert pc to grid
        # R_pc: point cloud value with [batch,channels,N] dim
        # XY_pc: point cloud position with [batch,2,N] dim
        #        scaled to [0,1]
        batch = 1
        _,height,width = self.XY_grd.shape

        XY_grd_tmp = self.XY_grd.reshape(batch,2,height*width).transpose(0,2,1)
        XY_pc_exp = np.expand_dims(np.stack([self.x_pc,self.y_pc], axis=0), axis=0)
        XY_pc_tmp = XY_pc_exp.transpose(0,2,1)
        R_pc_exp = np.expand_dims(self.R_pc,axis=[0,1])
        # interpolate
        R_grd_interp = nearest_neighbor_interp_kd(XY_grd_tmp,XY_pc_tmp,R_pc_exp)
        R_grd_interp = R_grd_interp.reshape(height,width)
        return R_grd_interp
    
    def l2_norm(self,V1,V2):
        return  np.sum(np.power((V1-V2),2))

    def reward_nearest_neighbor(self):
        """
        Calculate reward based on estimation error of true meteorological field
        from point observation with the nerest neighbor interpolation

        Returns:
            int: The reward for the current state.
            bool: Whether the episode is finished.
        """
        
        self.R_grd_interp = self.pc_to_grid_nearest()
        # The negative sign is used to work as a reward instead of a loss
        reward = -1.0*self.l2_norm(self.R_grd,self.R_grd_interp)

        done = False
        return reward,done

    def step(self,action):
        """
        Perform a single step of the environment given the action.
        
        Args:
            action (np.array): Action to be performed.
            
        Returns:
            tuple: Contains observation, reward, done flag, and an empty info dictionary.
        """
        # get next state given action

        # extract xy from state
        XY_pc = self.state[0:2*self.num_bots]

        # calculate next position given velocity and acceleration
        self.velocity = self.velocity + action * self.dt / 10000.0
        XY_pc = XY_pc + self.velocity * self.dt
        # clip xy position within arena area
        XY_pc = np.clip(XY_pc,-self.arena_size/2,self.arena_size/2)
        self.x_pc = XY_pc.reshape([2,self.num_bots])[0,:]
        self.y_pc = XY_pc.reshape([2,self.num_bots])[1,:]

        # get observation
        self.R_pc = self.grid_to_pc_nearest().flatten()

        # get next state
        self.state = np.concatenate([XY_pc,self.R_pc])

        # get obs
        obs = self.state

        # calculate reward
        reward,done = self.reward_nearest_neighbor()

        # if max step is reached, set the done flag
        self.steps += 1
        if self.steps > self._max_episode_steps:
            done = True

        print("steps:",self.steps," reward:",reward," done:",done)

        # reset the step
        if done == True:
            self.steps=0

            
        info = {}

        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        """
        Render the environment as an RGB array.
        
        Args:
            mode (str, optional): The rendering mode. Defaults to 'rgb_array'.
            
        Returns:
            np.array: RGB array of the environment.
        """
    
        # Plot obsbot positions
        self.plot_agent()
    
        # Convert matplotlib figure to RGB array
        rgb_array = self.fig2array()[:, :, :3]
    
        # Retun RGB array
        return rgb_array 
    
    def plot_agent(self,vmin=0,vmax=1):
        """
        Plot positions of observation bots and estimated meteo field 
        in the arena using matplotlib.
        """

        # clip xy in [0,1] range
        x_scaled = (self.x_pc / self.arena_size) + 0.5
        y_scaled = (self.y_pc / self.arena_size) + 0.5
        x_plt = np.clip(x_scaled,0,1)*self.R_grd_interp.shape[0]
        y_plt = np.clip(y_scaled,0,1)*self.R_grd_interp.shape[1]

        #self.fig = plt.figure(figsize=(7, 10), dpi=200)
        self.fig, self.ax = plt.subplots(2, 1, figsize=(7, 7))

        #self.ax = plt.axes()
        reward,done = self.reward_nearest_neighbor()
        self.fig.suptitle("Reward: %f" % reward)

        # 1st plot: ground truth
        #self.ax = self.fig.add_subplot(1, 2, 1)
        aximg = self.ax[0].imshow(self.R_grd,vmin=vmin,vmax=vmax,cmap="GnBu",origin='lower')
        self.fig.colorbar(aximg,ax=self.ax[0])
        self.ax[0].scatter(x_plt, y_plt, c=self.R_pc, cmap="GnBu", edgecolors="black")
        # set axes range
        self.ax[0].set_xlim(0, self.R_grd.shape[0])
        self.ax[0].set_ylim(0, self.R_grd.shape[1])
        self.ax[0].grid()  

        # 2nd plot: prediction
        #self.ax = self.fig.add_subplot(1, 2, 2)
        aximg2 = self.ax[1].imshow(self.R_grd_interp,vmin=vmin,vmax=vmax,cmap="GnBu",origin='lower')
        self.fig.colorbar(aximg2,ax=self.ax[1])
        self.ax[1].scatter(x_plt, y_plt, c=self.R_pc, cmap="GnBu", edgecolors="black")
        # set axes range
        self.ax[1].set_xlim(0, self.R_grd_interp.shape[0])
        self.ax[1].set_ylim(0, self.R_grd_interp.shape[1])
        self.ax[1].grid() 

        #self.fig.subplots_adjust(right=0.95)
        #cbar_ax = self.fig.add_axes([0.96, 0.15, 0.01, 0.7])
        #self.fig.colorbar(aximg,ax=cbar_ax)
    
    def fig2array(self):
        """
        Convert matplotlib figure to an RGB array.
        
        Returns:
            np.array: The RGB array representation of the matplotlib figure.
        """
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf = np.fromstring(self.fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        return buf
