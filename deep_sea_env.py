import random as rd
import torch

class DeepSea:
    def __init__(self, depth, noise_probability=0):
        """
        Initializes the DeepSea environment.
        Args:
            depth (int): The number of actions required to reach the bottom of the sea.
        Attributes:
            device (torch.device): The device (CPU or CUDA) used for computations.
            depth (int): The depth of the environment, representing the number of steps to reach the goal.
            state: The current state of the environment (initialized as None).
            done (torch.Tensor): A tensor indicating whether the episode is finished (0 for not done).
            actions (List[int]): A list of random actions (0 for left, 1 for right) for each step in the environment.
        Notes:
            - Action 0 corresponds to moving left, and action 1 corresponds to moving right.
        """
        self.noise_probability = noise_probability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth = depth
        self.state = None
        self.done = torch.tensor(0,dtype=torch.float32,device=self.device)
        self.actions = [ rd.randint(0, 1) for _ in range(depth) ]
    
    def coord_to_state(self, coord):
        """
        Converts a coordinate tuple into a one-hot encoded state tensor.
        Args:
            coord (tuple): A tuple (row, column) representing the coordinates in the environment.
        Returns:
            torch.Tensor: A tensor of shape (depth, depth) with a 1 at the specified coordinate,
                          or a zero tensor if the column index equals self.depth.
        """
        state = torch.zeros(self.depth,self.depth,device=self.device)
        if coord[1] == self.depth:
            return state
        else:
            state[coord[0], coord[1]] = 1
            return state

    def reset(self):
        """
        Resets the environment to the initial state.
        Returns:
            torch.Tensor: The encoded initial state after reset.
        """
        self.state = (0,0)
        self.done = torch.tensor(0,dtype=torch.float32,device=self.device)
        return self.coord_to_state(self.state)
    
    def step(self, action):
        """
        Performs a step in the DeepSea environment based on the given action.
        Args:
            action: The action to take at the current state.
        Returns:
            next_state: The encoded next state after taking the action.
            reward (torch.Tensor): The reward received after taking the action.
            done (torch.Tensor): A flag indicating whether the episode has ended.
        Notes:
            - The state is updated based on the action and current position.
            - The episode ends when the agent reaches the maximum depth.
            - A reward of 1 is given only if the agent reaches the "bottom-right" corner; otherwise, the reward is 0.
        """
        if self.noise_probability > 0:
            # Randomly change action with a probability of noise_probability
            if rd.random() < self.noise_probability:
                action = 1 - action
        # Update state
        if self.state[0] == 0 or action != self.actions[self.state[1]]:
            self.state = (self.state[0] + 1, self.state[1] + 1)
        else:
            self.state = (self.state[0]-1, self.state[1] + 1)
        # Update done
        if self.state[1] == self.depth:
            self.done = torch.tensor(1,dtype=torch.float32,device=self.device)
        # Compute reward
        if self.state[1] == self.depth and self.state[0] == self.depth:
            reward = torch.tensor(1,dtype=torch.float32,device=self.device)
        else :
            reward = torch.tensor(0,dtype=torch.float32,device=self.device)
        return self.coord_to_state(self.state), reward, self.done

    def render(self):
        """
        Renders the current state of the environment as a grid to the console.
        The agent's position is marked with 'X', and all other positions are marked with '.'.
        The grid has dimensions (depth+1) x (depth+1), where 'depth' is an attribute of the environment.
        """
        state = torch.zeros(self.depth+1, self.depth+1)
        state[self.state[0], self.state[1]] = 1
        for y in range(self.depth+1):
            for x in range(self.depth+1):
                if state[x,y] == 1:
                    print("X", end=" ")
                else:
                    print(".", end=" ")
            print()