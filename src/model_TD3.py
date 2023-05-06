
class ReplayBuffer(object):
    """ReplayBuffer class to store and sample experiences for training deep reinforcement learning algorithms."""

    def __init__(self, max_size=1e6):
        """
        Initialize the ReplayBuffer with the specified maximum size.
        
        Args:
            max_size (int, optional): Maximum number of transitions to store. Defaults to 1e6.
        """
        self.storage = []
        self.max_size = max_size
        self.ptr = 0  # ptr: pointer

    def add(self, transition):
        """
        Add a new transition to the ReplayBuffer.
        
        If the storage is full, overwrite the oldest transition.
        
        Args:
            transition (tuple): A tuple containing state, next_state, action, reward, and done.
        """
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size  # if the index is larger than max index, restart from zero
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        """
        Sample a random batch of transitions from the ReplayBuffer.
        
        Args:
            batch_size (int): Number of transitions to sample.
        
        Returns:
            tuple: A tuple containing batch_states, batch_next_states, batch_actions, batch_rewards, and batch_dones.
        """
        # select random batch from replay buffer
        ind = np.random.randint(0, len(self.storage), batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

class Actor(nn.Module):
    """
    Neural Network for the Actor Model / Actor Target.
    
    This class represents the Actor network in an Actor-Critic architecture for deep reinforcement learning.
    """

    def __init__(self, state_dim, action_dim, max_action):
        """
        Initialize the Actor network with the given state, action dimensions, and maximum action.
        
        Args:
            state_dim (int): Dimension of the input state vector.
            action_dim (int): Dimension of the output action vector.
            max_action (float): Maximum action value that can be taken.
        """
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        """
        Forward pass through the Actor network.
        
        Args:
            x (torch.Tensor): Input state vector.
        
        Returns:
            torch.Tensor: Output action vector.
        """
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x

class Critic(nn.Module):
    """
    Neural Network for the two Critic Models and the two Critic Targets.
    
    This class represents the Critic network in an Actor-Critic architecture for deep reinforcement learning.
    The Critic network has two separate neural networks that return Q values.
    """

    def __init__(self, state_dim, action_dim):
        """
        Initialize the Critic network with the given state and action dimensions.
        
        Args:
            state_dim (int): Dimension of the input state vector.
            action_dim (int): Dimension of the input action vector.
        """
        super(Critic, self).__init__()
        # Define the 1st critic neural network
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)

        # Define the 2nd critic neural network
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)

    def forward(self, x, u):
        """
        Forward pass through the Critic network.
        
        Args:
            x (torch.Tensor): Input state vector.
            u (torch.Tensor): Input action vector.
        
        Returns:
            tuple: Q values from the two separate critic neural networks.
        """
        xu = torch.cat([x, u], 1)

        # 1st Critic
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)

        # 2nd Critic
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)

        return x1, x2

    def Q1(self, x, u):
        """
        Compute Q value using the first critic neural network.
        
        Args:
            x (torch.Tensor): Input state vector.
            u (torch.Tensor): Input action vector.
        
        Returns:
            torch.Tensor: Q value from the first critic neural network.
        """
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

class TD3(object):
    """
    Building the whole Training Process into a class.
    
    This class represents the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm for deep reinforcement learning.
    It includes the training process and methods for saving and loading the model.
    """

    def __init__(self, state_dim, action_dim, max_action):
        """
        Initialize the TD3 algorithm with the given state, action dimensions, and maximum action.
        
        Args:
            state_dim (int): Dimension of the input state vector.
            action_dim (int): Dimension of the output action vector.
            max_action (float): Maximum action value that can be taken.
        """
        # Define Actor networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        # Initialize actor target with actor weights
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Define Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        # Initialize actor target with critic weights
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action

    def select_action(self, state):
        """
        Select an action based on the given state.
        
        Args:
            state (np.array): Input state vector.
        
        Returns:
            np.array: Output action vector.
        """
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        """
        Train the TD3 model.
        
        Args:
            replay_buffer (ReplayBuffer): Replay buffer to sample transitions from.
            iterations (int): Number of training iterations.
            batch_size (int, optional): Batch size for training.
            discount (float, optional): Discount factor (gamma).
            tau (float, optional): Polyak averaging coefficient.
            policy_noise (float, optional): Gaussian noise added to the action for exploration. 
            noise_clip (float, optional): Maximum allowed value for the Gaussian noise. 
            policy_freq (int, optional): Frequency of Actor network updates.
        """
        for it in range(iterations):
            # Sample a batch of transitions(s,s',a,r) from the replay-buffer memory
            batch_states,batch_next_states,batch_actions,batch_rewards,batch_dones = replay_buffer.sample(batch_size)
            # convert to torch.tensor
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)
            
            # From the next state s', the actor target plays the next action a'
            next_action = self.actor_target(next_state)
            
            # Add Gaussian noise next action and clip the value
            noise = torch.Tensor(batch_actions).data.normal_(0,policy_noise).to(device)
            noise = noise.clamp(-noise_clip,noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action,self.max_action)
            
            # The two critic targes take (s', a') as input
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            
            # Take the minimum of Q1 and Q2
            # This is for avoiding critic to produce overly optimistic values
            target_Q = torch.min(target_Q1, target_Q2)
            
            # Final tatrget of the two Critic models: Qt = r + gamma * min(Q1,Q2)
            # where gamma is the discount factor
            # multiply by "done" to use only reward if the episode is finished
            
            # "detach" is necessary, since target_Q is not going to be used for backprop.
            target_Q = reward + ((1 - done) * discount * target_Q).detach()
            
            # The two critic models take (s,a) and calculate Q
            current_Q1, current_Q2 = self.critic(state, action)
            
            # Compute the loss coming from the two critic models.
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            # Backprop Critic loss and update the parameters of the two critic models.
            # Note that "critic target" is NOT updated by backprop, since it is updated by temporal smoothing of critic.
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step() # update the weights
            
            # Once every two iterations ("D" of DDPG), backprop Actor model by gradient ascent
            # Q value is an approximate return.
            if it % policy_freq == 0:
                actor_loss = - self.critic.Q1(state, self.actor(state)).mean()# negative sign is for maximization
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step() # update the weights
            
                # Update actor target by Polyak averaging
                for param, target_param in zip(self.actor.parameters(),self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1-tau)*target_param.data)
              
                # Update critic targets by Polyak averaging
                for param, target_param in zip(self.critic.parameters(),self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1-tau)*target_param.data)            # Sample a batch of transitions from the replay buffer
                    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            
def save(self, filename, directory):
    """
    Save the trained model.
    
    Args:
        filename (str): Filename for the saved model.
        directory (str): Directory where the model will be saved.
    """
    torch.save(self.actor.state_dict(),  '%s/%s_actor.pth'  % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

def load(self, filename, directory):
    """
    Load a pre-trained model.
    
    Args:
        filename (str): Filename of the saved model.
        directory (str): Directory where the model is located.
    """
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
