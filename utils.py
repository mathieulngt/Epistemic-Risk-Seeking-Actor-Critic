import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import random as rd
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import os
from datetime import datetime
import json
import pickle

def set_seed(seed):
    np.random.seed(seed)
    rd.seed(seed)
    torch.manual_seed(seed)

def plot_policy(policy, env):
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
        plt.imshow(policy_arr.transpose(0, 1).numpy(), cmap="gray_r", vmin=0, vmax=1)
        plt.show()

def plot_policy_shared(shared_body,policy_head, env):
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
        plt.imshow(policy_arr.transpose(0, 1).numpy(), cmap="gray_r", vmin=0, vmax=1)
        plt.show()

def plot_policy_saving(policy, env, path):
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
        plt.imshow(policy_arr.transpose(0, 1).numpy(), cmap="gray_r", vmin=0, vmax=1)
        plt.savefig(path)
        plt.close()
     
def plot_rewards_and_variances(rewards, variances, window=100, title="Évolution des rewards et variances"):
    """
    Affiche deux sous-figures :
    - Les rewards lissés par une moyenne glissante (axe x : épisodes)
    - Les variances lissées par une moyenne glissante (axe x : steps)
    """
    if len(rewards) < window or len(variances) < window:
        print("Pas assez de données pour lisser.")
        return

    smoothed_rewards = [np.mean(rewards[max(0, i - window + 1):i + 1]) for i in range(len(rewards))]
    smoothed_variances = [np.mean(variances[max(0, i - 5 + 1):i + 1]) for i in range(len(variances))]

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Reward plot (par épisode)
    axs[0].plot(smoothed_rewards, label=f'Rewards lissés ({window} épisodes)', color='tab:blue')
    axs[0].set_ylabel('Reward')
    axs[0].set_xlabel('Épisode')
    axs[0].legend()
    axs[0].set_title(title)

    # Variance plot (par step)
    axs[1].plot(smoothed_variances, label=f'Variances lissées ({5} steps)', color='tab:orange')
    axs[1].set_ylabel('Variance')
    axs[1].set_xlabel('Step')
    axs[1].legend()
    axs[1].set_title("Évolution des variances par step")

    plt.tight_layout()
    plt.show()

def save_plot_rewards_and_variances(rewards, variances, taus=None, entropies=None ,window=100, title=None,path=None):
    """
    Affiche jusqu'à trois sous-figures :
    - Rewards lissés par une moyenne glissante (axe x : épisodes)
    - Variances lissées par une moyenne glissante (axe x : steps)
    - Valeurs de tau par épisode (optionnel)
    """
    # Détache les tenseurs PyTorch si nécessaire
    rewards = [r.item() if isinstance(r, torch.Tensor) else r for r in rewards]
    variances = [v.item() if isinstance(v, torch.Tensor) else v for v in variances]
    if taus is not None:
        taus = [t.item() if isinstance(t, torch.Tensor) else t for t in taus]

    if entropies is not None:
        entropies = [e.item() if isinstance(e, torch.Tensor) else e for e in entropies]

    if len(rewards) < window or len(variances) < 5 or (taus is not None and len(taus) < 5):
        print("Pas assez de données pour lisser.")
        return

    window_var = 1
    window_tau = 1
    window_entropies = 1
    smoothed_rewards = [np.mean(rewards[0:i + 1]) for i in range(len(rewards))]
    smoothed_variances = [np.mean(variances[max(0, i - window_var + 1):i + 1]) for i in range(len(variances))]
    if taus is not None:
        smoothed_taus = [np.mean(taus[max(0, i - window_tau + 1):i + 1]) for i in range(len(taus))]
    if entropies is not None:
        smoothed_entropies = [np.mean(entropies[max(0, i -  window_entropies + 1):i + 1]) for i in range(len(entropies))]

    n_plots = 2
    if taus is not None:
        n_plots+=1
    if entropies is not None:
        n_plots+=1
    
    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))

    # Reward plot (par épisode)
    axs[0].plot(smoothed_rewards, label="Success rate", color='black') #label=f'Rewards lissés ({window} épisodes)'
    #axs[0].set_ylabel('Success rate')
    axs[0].set_xlabel('Episode')
    axs[0].legend()
    #axs[0].set_title(title)

    # Variance plot (par step)
    axs[1].plot(smoothed_variances, label=f'Episode mean uncertainty', color='tab:red')
    #axs[1].set_ylabel('Episode mean uncertainty')
    axs[1].set_xlabel('Episode')
    axs[1].legend()
   #axs[1].set_title("Evolution of episode uncertainty during training")

    # Tau plot (par épisode)
    if taus is not None:
        axs[2].plot(smoothed_taus, label=f'Tau', color='tab:blue')
        #axs[2].set_ylabel('Tau')
        axs[2].set_xlabel('Episode')
        axs[2].legend()
        #axs[2].set_title("Evolution of Tau during training")

    if entropies is not None:
        axs[3].plot(smoothed_entropies, label=f'Episode mean entropy', color='tab:green')
        #axs[3].set_ylabel('Episode mean entropy')
        axs[3].set_xlabel('Episode')
        axs[3].legend()
        #axs[3].set_title("Evolution of cumulative entropy during training")

    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close()

def plot_smoothed_rewards_saving(path,rewards, env, window=100):
    """
    Affiche la courbe des rewards lissée par une moyenne glissante sur 'window' épisodes.
    """
    if len(rewards) < window:
        print("Pas assez de données pour lisser.")
        return
    smoothed = [np.mean(rewards[max(0, i-window+1):i+1]) for i in range(len(rewards))]
    plt.figure(figsize=(8, 4))
    plt.plot(smoothed, label=f'Moyenne glissante ({window} épisodes)')
    plt.xlabel('Épisode')
    plt.ylabel('Reward cumulée lissée')
    plt.title(f'Depth = {env.depth} - Évolution des rewards cumulées (moyenne glissante)')
    plt.legend()
    plt.savefig(path)
    plt.close()

def plot_tensor_triplets(data):
    """
    Plot a list of 5-uplets (t1, t2, t3, date1, date2) as individual figures.
    Each figure has 3 subplots (one per tensor), and is titled with the two dates.
    
    Parameters:
    data (list of tuples): Each tuple contains (t1, t2, t3, date1, date2),
                           where t1, t2, t3 are 2D tensors (numpy arrays or torch tensors),
                           and date1/date2 are strings.
    """
    for i, (t1, t2, t3, date1, date2) in enumerate(data):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Dates : {date1} - {date2}", fontsize=16)
        
        for ax, tensor, label in zip(axes, [t1, t2, t3], ['Critic J', 'Critic K', 'Policy']):
            if label == "Policy":
                im = ax.imshow(tensor, aspect='auto',  cmap="gray_r", vmin=0, vmax=1)
            else:
                im = ax.imshow(tensor, aspect='auto') # map="Blues", vmin=0, vmax=2
            ax.set_title(label)
            fig.colorbar(im, ax=ax)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def plot_k_j_pi_counts(data):
    """
    Plot a list of 5-uplets (t1, t2, t3,t4,t5 date1) as individual figures.

    Each figure has 5 subplots (one per tensor), and is titled with the two dates.
    
    Parameters:
    data (list of tuples): Each tuple contains (t1, t2, t3,t4,t5, date1),
                           where t1, t2, t3, t4 are 2D tensors (numpy arrays or torch tensors),
                           and date1 is a string.
    """
    for i, (t1, t2, t3,t4,t5,date1) in enumerate(data):
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        fig.suptitle(f"Episode : {date1}", fontsize=16)

        for ax, tensor, label in zip(axes, [t1, t2, t3,t4,t5], ['Critic J', 'Critic K (True)', "Critic K (False)" , 'Policy', 'Counts']):
            if label == "Policy":
                im = ax.imshow(tensor, aspect='auto',  cmap="gray_r", vmin=0, vmax=1)
            else:
                im = ax.imshow(tensor, aspect='auto')
            ax.set_title(label)
            fig.colorbar(im, ax=ax)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def plot_k_j_pi_counts_taus(data):
    """
    Plot a list of 6-uplets (t1, t2, t3, t4, t5, t6, date1) as individual figures.

    Each figure has 6 subplots (one per tensor), and is titled with the episode/date.

    Parameters:
    data (list of tuples): Each tuple contains (t1, t2, t3, t4, t5, t6, date1),
                           where t1 to t6 are 2D tensors (numpy arrays or torch tensors),
                           and date1 is a string.
    """
    for i, (t1, t2, t3, t4, t5, t6, date1) in enumerate(data):
        fig, axes = plt.subplots(1, 6, figsize=(24, 6))
        fig.suptitle(f"Episode : {date1}", fontsize=16)

        for ax, tensor, label in zip(
            axes, 
            [t1, t2, t3, t4, t5, t6], 
            ['Critic J', 'Critic K (True)', 'Critic K (False)', 'Policy', 'Counts', 'Taus']
        ):
            if label == "Policy":
                im = ax.imshow(tensor, aspect='auto', cmap="gray_r", vmin=0, vmax=1)
            else:
                im = ax.imshow(tensor, aspect='auto')
            ax.set_title(label)
            fig.colorbar(im, ax=ax)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def plot_tensor_quad(data):

    """
    Plot a list of 5-uplets (t1, t2, t3, date1, date2) as individual figures.
    Each figure has 3 subplots (one per tensor), and is titled with the two dates.
    
    Parameters:
    data (list of tuples): Each tuple contains (t1, t2, t3, date1, date2),
                           where t1, t2, t3 are 2D tensors (numpy arrays or torch tensors),
                           and date1/date2 are strings.
    """
    for i, (t1, t2, t3, t4, t5,date1, date2) in enumerate(data):
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        fig.suptitle(f"Dates : {date1} - {date2}", fontsize=16)
        
        for ax, tensor, label in zip(axes, [t1, t2, t3, t4,t5], ['Critic J', 'Critic K', 'Critic K - J' ,'Policy',"Counts"]):
            if label == "Policy":
                im = ax.imshow(tensor, aspect='auto',  cmap="gray_r", vmin=0, vmax=1)
            else:
                im = ax.imshow(tensor, aspect='auto') # map="Blues", vmin=0, vmax=2
            ax.set_title(label)
            fig.colorbar(im, ax=ax)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def plot_seen_pair(data,x,y):
    t = torch.stack(data)
    t = t[:,x,y,0] + t[:,x,y,1] 
    plt.figure(figsize=(10, 5))
    plt.plot(t.numpy(), label=f'Pair ({x}, {y})')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    plt.title(f'Seen Pair ({x}, {y})')
    plt.legend()
    plt.show()

def save_data(data, base_dir):
    # Crée le répertoire de sauvegarde (ex: saves/exp42)
    os.makedirs(base_dir, exist_ok=True)

    # Sauvegarde complète (pickle)
    # with open(os.path.join(base_dir, "slow.pkl"), 'wb') as f:
    #     pickle.dump(data, f)

    # Sauvegarde allégée (JSON)
    restricted_data = {key: data[key] for key in [
        "hyperparameters",
        "episodes to reach 90% success",
        "episodes to reach first reward",
        "episodes to reach 10% success"
    ] if key in data}  # <- évite KeyError

    with open(os.path.join(base_dir, "fast.json"), 'w') as f:
        json.dump(restricted_data, f, indent=4)