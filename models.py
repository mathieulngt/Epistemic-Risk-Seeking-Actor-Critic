import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self,env,N_size=200):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(env.depth*env.depth, N_size),
            nn.ReLU(),
            nn.Linear(N_size, 2)
        )

    def forward(self, x):
        return self.network(x)
    
class Policy(nn.Module):
    def __init__(self,env,N_size=200):
        super(Policy, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(env.depth*env.depth, N_size),
            nn.ReLU(),
            nn.Linear(N_size, 2)
        )

    def forward(self, x):
        return F.softmax(self.network(x), dim=-1)
    
class Critic_J(nn.Module):
    def __init__(self,env,N_size=200):
        super(Critic_J, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(env.depth*env.depth, N_size),
            nn.ReLU(),
            nn.Linear(N_size, 1)
        )

    def forward(self, x):
        return self.network(x)

class Reward_head(nn.Module):
    def __init__(self,env,n_mid):
        super(Reward_head,self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(env.depth*env.depth, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid,2)
        )
        self.prior = nn.Sequential(
            nn.Flatten(),
            nn.Linear(env.depth*env.depth, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid,2)
        )
        for param in self.prior.parameters():
            param.requires_grad = False
    
    def forward(self,x):
        return self.network(x) + self.prior(x)

class Reward(nn.Module):
    def __init__(self,env,n_heads,n_mid):
        super(Reward,self).__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([ Reward_head(env,n_mid) for _ in range(n_heads) ])

    def forward(self,x):
        """
        Forward pass of the reward model.

        Args:
            x (torch.Tensor): the input of the model, of shape (B, H, W)

        Returns:
            list of torch.Tensor: the rewards for each head, of shape (B, 1)
        """
        return [ self.heads[i](x) for i in range(self.n_heads) ]
    
class Shared_body(nn.Module):
    def __init__(self,env,N_size=200):
        super(Shared_body, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(env.depth*env.depth, N_size),
            nn.ReLU()
        )
    def forward(self, x):
        return self.network(x)
    
class Policy_head(nn.Module):
    def __init__(self,env,N_size=200):
        super(Policy_head, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(N_size, 2)
        )

    def forward(self, x):
        return F.softmax(self.network(x), dim=-1)

class Critic_k_head(nn.Module):
    def __init__(self,env,N_size=200):
        super(Critic_k_head, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(N_size, 2)
        )

    def forward(self, x):
        return self.network(x)
    
class Critic_j_head(nn.Module):
    def __init__(self,env,N_size=200):
        super(Critic_j_head, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(N_size, 1)
        )

    def forward(self, x):
        return self.network(x)

def make_fig_policy(env,policy,ax):
    policy_arr = torch.zeros(env.depth, env.depth)
    for i in range(env.depth):
        for j in range(env.depth):
            if (i <= j):
                state = env.coord_to_state((i,j))
                with torch.no_grad():
                    logits = policy(state.unsqueeze(0))
                    policy_arr[i,j] = logits[0][1]
                    if env.actions[j] == 1:
                        policy_arr[i,j] = 1 - policy_arr[i,j]
    policy_arr[0,0] = 1  # Initial state is always 1
    ax.imshow(policy_arr.transpose(0, 1).numpy(), cmap="gray_r", vmin=0, vmax=1)

def make_fig_policy_shared(env, shared_body, policy_head, ax, title=None):
    policy_arr = torch.zeros(env.depth, env.depth)
    for i in range(env.depth):
        for j in range(env.depth):
            if i <= j:
                state = env.coord_to_state((i, j))
                with torch.no_grad():
                    logits = policy_head(shared_body(state.unsqueeze(0)))
                    policy_arr[i, j] = logits[0][1]
                    if env.actions[j] == 1:
                        policy_arr[i, j] = 1 - policy_arr[i, j]
    policy_arr[0, 0] = 1  # Initial state is always 1

    # Affichage de la matrice avec les indices comme axes
    im = ax.imshow(policy_arr.transpose(0, 1).numpy(), cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks(range(env.depth))
    ax.set_yticks(range(env.depth))

    if title:
        ax.set_title(title)

def make_fig_rewards(env, reward_ensemble, ax,title=None):
    rewards_arr = torch.zeros(env.depth, env.depth)
    for i in range(env.depth):
        for j in range(env.depth):
            if i <= j:
                state = env.coord_to_state((i, j))
                with torch.no_grad():
                    estimated_reward = torch.stack(reward_ensemble(state.unsqueeze(0))).squeeze(1)
                    std_per_action = estimated_reward.std(dim=0)  
                    if env.actions[j] == 1:
                        rewards_arr[i, j] = std_per_action[0]
                    else:
                        rewards_arr[i, j] = std_per_action[1]

    # Affichage de la matrice avec les indices comme axes
    im = ax.imshow(rewards_arr.transpose(0, 1).numpy(), cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks(range(env.depth))
    ax.set_yticks(range(env.depth))

    if title:
        ax.set_title(title)

def make_fig_critic_k(env, shared_body,critic_k_head, ax,title=None):
    estimates_arr = torch.zeros(env.depth, env.depth)
    for i in range(env.depth):
        for j in range(env.depth):
            if i <= j:
                state = env.coord_to_state((i, j))
                with torch.no_grad():
                    estimates = critic_k_head(shared_body(state.unsqueeze(0))).squeeze(0)
                    if env.actions[j] == 1:
                        estimates_arr[i, j] = estimates[0]
                    else:
                        estimates_arr[i, j] = estimates[1]

    # Affichage de la matrice avec les indices comme axes
    im = ax.imshow(estimates_arr.transpose(0, 1).numpy(), cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks(range(env.depth))
    ax.set_yticks(range(env.depth))

    if title:
        ax.set_title(title)

def make_fig_critic_j(env, shared_body,critic_j_head, ax,title=None):
    estimates_arr = torch.zeros(env.depth, env.depth)
    for i in range(env.depth):
        for j in range(env.depth):
            if i <= j:
                state = env.coord_to_state((i, j))
                with torch.no_grad():
                    estimates = critic_j_head(shared_body(state.unsqueeze(0))).squeeze(0)
                    estimates_arr[i, j] = estimates[0]

    # Affichage de la matrice avec les indices comme axes
    im = ax.imshow(estimates_arr.transpose(0, 1).numpy(), cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks(range(env.depth))
    ax.set_yticks(range(env.depth))

    if title:
        ax.set_title(title)

def critic_j_shared_to_tensor(env, shared_body,critic_j_head):
    estimates_arr = torch.zeros(env.depth, env.depth)
    for i in range(env.depth):
        for j in range(env.depth):
            if i <= j:
                state = env.coord_to_state((i, j))
                with torch.no_grad():
                    estimates = critic_j_head(shared_body(state.unsqueeze(0))).squeeze(0)
                    estimates_arr[i, j] = estimates[0]
    return estimates_arr.transpose(0, 1)

def critic_k_shared_to_tensor(env, shared_body,critic_k_head):
    estimates_arr_true = torch.zeros(env.depth, env.depth)
    estimates_arr_false = torch.zeros(env.depth, env.depth)
    for i in range(env.depth):
        for j in range(env.depth):
            if i <= j:
                state = env.coord_to_state((i, j))
                with torch.no_grad():
                    estimates = critic_k_head(shared_body(state.unsqueeze(0))).squeeze(0)
                    if env.actions[j] == 1:
                        estimates_arr_true[i, j] = estimates[0]
                        estimates_arr_false[i, j] = estimates[1]
                    else:
                        estimates_arr_true[i, j] = estimates[1]
                        estimates_arr_false[i, j] = estimates[0]

    return estimates_arr_true.transpose(0, 1), estimates_arr_false.transpose(0, 1)

def policy_shared_to_tensor(env, shared_body,policy_head):
    policy_arr = torch.zeros(env.depth, env.depth)
    for i in range(env.depth):
        for j in range(env.depth):
            if (i <= j):
                state = env.coord_to_state((i,j))
                with torch.no_grad():
                    logits = policy_head( shared_body(state.unsqueeze(0)))
                    policy_arr[i,j] = logits[0][1]
                    if env.actions[j] == 1:
                        policy_arr[i,j] = 1 - policy_arr[i,j]
    policy_arr[0,0] = 1  # Initial state is always 1
    return policy_arr.transpose(0, 1)

def reward_ensemble_to_tensor(env, uncertainty_scale,reward_ensemble):
    stds = torch.zeros(env.depth, env.depth)
    for i in range(env.depth):
        for j in range(env.depth):
            if (i <= j):
                state = env.coord_to_state((i,j))
                with torch.no_grad():
                    estimated_reward = torch.stack(reward_ensemble(state.unsqueeze(0)))
                    estimated_reward_0 = estimated_reward[:,0,0]
                    estimated_reward_1 = estimated_reward[:,0,1]
                    pow = 2
                    uncs0 = (uncertainty_scale * (torch.std(estimated_reward_0)** (pow) ))
                    uncs1 = (uncertainty_scale * (torch.std(estimated_reward_1)** (pow) ))
                    bound = 0.000001
                    stds[i,j] = 1/max(uncs0,bound) + 1/max(uncs1,bound)
    return stds
