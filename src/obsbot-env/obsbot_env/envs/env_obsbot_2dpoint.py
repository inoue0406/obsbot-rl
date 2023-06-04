"""
ObsBot2DPoint class for simulating a 2D environment with observation bots.
This class inherits from gym.Env and defines methods to simulate the 2d point observation,
such as rainfall or temperature gauges given a 2d meteorological field.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import random
import gym

# import interpolator
from obsbot_env.envs.nearest_neighbor_interp_kdtree import nearest_neighbor_interp_kd

class ObsBot2DPoint(gym.Env):
    def __init__(self, num_bots):
        """
        Initialize the environment with the given number of observation bots.
        
        Args:
            num_bots (int): Number of observation bots.
        """
        super(ObsBot2DPoint, self).__init__()

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
        # (X,Y) position is taken as observation
        # * This is only for a test run, and this is to be changed to observed meteorological field
        self.observation_space = gym.spaces.Box(-self.arena_size/2,self.arena_size/2, (2*self.num_bots,), np.float32)

        # Reward range
        self.reward_range = (0, 1000)

        # Grid size of a meteorological field given as the ground truth
        self.field_height = 200
        self.field_width = 200

    def reset(self):
        """
        Reset the environment to an initial state.
        
        Returns:
            np.array: The initial state of the environment.
        """
        # Start from arbitrary position
        self.state = np.random.randint(-self.arena_size/2,self.arena_size/2,2*self.num_bots)
        # Initialize with zero velocity
        self.velocity = np.zeros(2*self.num_bots)

        # Set xy positions of observation bots
        self.x_pc = self.state.reshape([2,self.num_bots])[0,:]
        self.y_pc = self.state.reshape([2,self.num_bots])[1,:]

        # Set 2d meteorological field
        self.R_grd = np.zeros((self.field_height,self.field_width))

        # Set xy coordinates for 2d grid field
        self.XY_grd = self.xy_grid(self.field_height, self.field_width)

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

        R_pc = self.obs
        XY_grd_tmp = self.XY_grd.reshape(batch,2,height*width).transpose(0,2,1)
        XY_pc_exp = np.expand_dims(np.stack([self.x_pc,self.y_pc], axis=0), axis=0)
        XY_pc_tmp = XY_pc_exp.transpose(0,2,1)
        # interpolate
        R_grd_interp = nearest_neighbor_interp_kd(XY_grd_tmp,XY_pc_tmp,R_pc)
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
        
        R_grd_interp = self.pc_to_grid_nearest()
        # The negative sign is used to work as a reward instead of a loss
        reward = -1.0*self.l2_norm(self.R_grd,R_grd_interp)

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

        # calculate next position given velocity and acceleration
        self.velocity = self.velocity + action * self.dt / 10000.0
        self.state = self.state + self.velocity * self.dt
        # clip xy position within arena area
        self.state = np.clip(self.state,-self.arena_size/2,self.arena_size/2)

        # get observation
        self.obs = self.grid_to_pc_nearest()

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

        return self.obs, reward, done, info

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
    
    def plot_agent(self):
        """
        Plot positions of observation bots in the arena using matplotlib.
        """
        self.fig = plt.figure(figsize=(7, 7), dpi=200)
        self.ax = plt.axes()
        #self.ax.axis("off")
        self.ax.set_xlim(-self.arena_size/2,self.arena_size/2)
        self.ax.set_ylim(-self.arena_size/2,self.arena_size/2)
        self.ax.grid()

        reward,done = self.reward_rect()
        self.ax.set_title("Reward: %f" % reward)

        # Plot goal area
        rect = patches.Rectangle(xy=(self.xreward1, self.yreward1), 
                                 width=self.xreward2-self.xreward1,
                                 height=self.yreward2-self.yreward1,
                                 fc='y')
        self.ax.add_patch(rect)
    
        xpos = self.state.reshape([2,self.num_bots])[0,:]
        ypos = self.state.reshape([2,self.num_bots])[1,:]
        scatter = self.ax.scatter(xpos,ypos,marker="o",c="blue",s=30)
    
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