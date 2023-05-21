"""
ObsBot2D class for simulating a 2D environment with observation bots.
This class inherits from gym.Env and defines methods to simulate the movement of observation bots in a rectangular arena.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import random
import gym

class ObsBot2D(gym.Env):
    def __init__(self, num_bots):
        """
        Initialize the environment with the given number of observation bots.
        
        Args:
            num_bots (int): Number of observation bots.
        """
        super(ObsBot2D, self).__init__()

        # This is to be referenced by Gym Wrappers.
        self.metadata = {'render.modes': ['rgb_array']}
        
        # speed limit for observation bots in [m/s]
        self.speed_limit = 1.0
        
        # define the size of a rectangular arena in [m]
        self.arena_size = 10000.0

        # define XY coordinates of the reward area
        self.xreward1 = 2000.0
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

    def reset(self):
        """
        Reset the environment to an initial state.
        
        Returns:
            np.array: The initial state of the environment.
        """
        # start from arbitrary position
        self.state = np.random.randint(-self.arena_size/2,self.arena_size/2,2*self.num_bots)
        return self.state

    def reward_rect(self):    
        """
        Calculate reward based on the positions of the bots in the arena.
        
        Returns:
            int: The reward for the current state.
            bool: Whether the episode is finished.
        """
        # give reward if Bots reach rightmost part of arena 
        x = self.state.reshape([2,self.num_bots])[0,:]
        y = self.state.reshape([2,self.num_bots])[1,:]
        count_right = (x > self.xreward1) &  (x < self.xreward2) &  (y > self.yreward1) &  (y < self.yreward2) 
        reward = sum(count_right)

        # finish if all the bots reach right most part of arena
        done = False
        if sum(count_right) == self.num_bots:
            done = True
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

        # calculate next position given velocity
        velocity = action
        dxy =  velocity * self.dt
        self.state = self.state + dxy
        # clip xy position within arena area
        self.state = np.clip(self.state,-self.arena_size/2,self.arena_size/2)

        # calculate reward
        reward,done = self.reward_rect()

        # if max step is reached, set the done flag
        self.steps += 1
        if self.steps > self._max_episode_steps:
            done = True

        print("steps:",self.steps," reward:",reward," done:",done)

        # reset the step
        if done == True:
            self.steps=0

        # set observation
        obs = self.state / self.scaling_coeff
            
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
