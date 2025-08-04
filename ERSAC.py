import matplotlib.pyplot as plt
import numpy as np
import random as rd
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from deep_sea_env import DeepSea
from models import Reward,Policy,Critic,Policy_head,Critic_k_head,Critic_j_head,Shared_body,make_fig_policy,make_fig_policy_shared,make_fig_rewards,make_fig_critic_k,make_fig_critic_j, critic_j_shared_to_tensor,critic_k_shared_to_tensor,policy_shared_to_tensor,reward_ensemble_to_tensor
import utils

import time 
from datetime import timedelta

def ersac(env,seed,
             shared_body,policy_head,critic_j_head,critic_k_head,reward_ensemble,
             learning_rate,gradient_clipping,Lambda,uncertainty_scale,mid_size,reward_estimation,n_heads,init_tau,min_tau,pow_seen_pairs,
             n_episode,
             early_stopping_first_reward=False,
             saving_path=None,
             LOGGING = False
             ):
    """
    Train the ERSAC (Epistemic Risk-Seeking Actor-Critic) algorithm.

    Parameters
    ----------
    env : object
        Environment object with `reset()`, `step(action)`, `depth`, `actions`, `noise_probability`, and `device` attributes.
    seed : int
        Random seed for reproducibility.
    shared_body : torch.nn.Module
        Shared feature extractor network used by both actor and critics.
    policy_head : torch.nn.Module
        Head network predicting action distributions (policy).
    critic_j_head : torch.nn.Module
        State-value critic network `J(s)` estimating expected return.
    critic_k_head : torch.nn.Module
        Action-value critic network `K(s, a)` estimating expected return for state-action pairs.
    reward_ensemble : torch.nn.Module
        Ensemble of networks to estimate the reward and its uncertainty (only used if `reward_estimation=True`).
    learning_rate : float
        Learning rate for shared networks and heads.
    gradient_clipping : float or None
        Maximum norm for gradient clipping (if None, no clipping is applied).
    Lambda : float
        Discount factor used in TD(λ) update for critic K.
    uncertainty_scale : float
        Scale applied to the intrinsic uncertainty bonus.
    mid_size : int
        Intermediate layer size in the network architectures (used for logging).
    reward_estimation : bool
        Whether to use a learned reward ensemble to compute uncertainty and guide exploration.
    n_heads : int
        Number of networks in the reward ensemble.
    init_tau : float
        Initial entropy regularization coefficient.
    min_tau : float
        Minimum allowed value for tau.
    pow_seen_pairs : float
        Exponent applied to the count of seen state-action pairs for scaling uncertainty.
    n_episode : int
        Maximum number of training episodes.
    early_stopping_first_reward : bool, optional
        Whether to stop training once the agent receives the first reward. Default is False.
    saving_path : str or None, optional
        If provided, training data and snapshots will be saved to this path. Default is None.
    LOGGING : bool, optional
        Whether to log snapshots of the policy and critics at regular intervals. Default is False.

    Returns
    -------
    training_data : dict
        Dictionary containing hyperparameters, performance metrics, logging data, 
        and timing information for the training process. Includes:
        - Cumulative rewards
        - Policy and critic snapshots
        - Uncertainty statistics
        - Time metrics
        - Episode statistics (first reward, success rate, etc.)

    Notes
    -----
    - This function includes internal logging and optional saving of the training process.
    - The reward ensemble, if enabled, is trained online using the true reward signal.
    - The entropy coefficient tau is updated dynamically based on the discrepancy between policy entropy and uncertainty.
    - The agent is considered "successful" in an episode if it collects a cumulative reward of 1.
    - The training stops early if 10% success rate is reached or `early_stopping_first_reward=True` and a reward is collected.
    """

    # initialize optimization structures
    learning_rate_reward_ensemble = 0.0005
    optimizer_shared = optim.RMSprop(
        [
            {"params": shared_body.parameters(), "lr": learning_rate/3},
            {"params": policy_head.parameters()},          
            {"params": critic_j_head.parameters()},
            {"params": critic_k_head.parameters()},
        ],
        lr=learning_rate 
    )
    optimizer_reward_ensemble = optim.RMSprop(reward_ensemble.parameters(), lr=learning_rate_reward_ensemble)
    criterion = nn.MSELoss()
    tau = init_tau
    step_bound_tau = 0.01
    gradient_clipping_reward_ensemble = None
    seen_pairs = torch.ones((env.depth, env.depth,2))

    # logging
    training_data = {}
    training_data["hyperparameters"] = {
        "depth" : env.depth,
        "seed":seed,
        "noise_probability": env.noise_probability,
        'learning_rate': learning_rate,
        "learning_rate_reward_ensemble" : learning_rate_reward_ensemble,
        "gradient_clipping" : gradient_clipping,
        "mid_size": mid_size,
        "lambda": Lambda,
        "pow_seen_pairs": pow_seen_pairs,
        "init_tau" : init_tau,
        "min_tau" : min_tau,
        "step_bound_tau" : step_bound_tau,
        "actions" : env.actions,
        "reward_estimation": reward_estimation,
        "n_heads": n_heads,
        "uncertainty_scale": uncertainty_scale,
        'n_episode': n_episode,
        "device":env.device.__str__(),
        "early_stopping_first_reward" : early_stopping_first_reward,
        "optimizer_shared": optimizer_shared.__class__.__name__,
        "optimizer_reward_ensemble": optimizer_reward_ensemble.__class__.__name__,
        "criterion": criterion.__class__.__name__,
        "criterion reward ensemble ": "mean absolute value",
        "gradient clipping reward ensemble" : gradient_clipping_reward_ensemble
    }
    training_data["episodes to reach 10% success"] = None
    training_data["episodes to reach first reward"] = None
    training_data["successful episodes"] = []
    training_data["cumulative_rewards"] = []
    training_data["actions history"] = []
    training_data["uncertainties"] = []
    training_data["episode cumulative uncertainty"] = []
    training_data["episode cumulative entropies"] = []
    training_data["taus"] = []
    training_data["seen_pairs history"] = []
    training_data["periodic snapshots"] = []
    training_data["exceptional snapshots"] = []
    training_data["uncertainty decrease"] = []
    success_count = 0
    periodic_logging = 100
    logging_interval = n_episode // 20
    print(training_data["hyperparameters"],flush=True)
    # time measurement
    training_data["episode_time"] = 0
    training_data["compute_loss_time"] = 0
    training_data["update_time"] = 0
    start_time_global = time.time() 

    if saving_path is not None:
        utils.save_data(training_data,saving_path)

    for episode in range(n_episode):
        # Initialize episode
        state = env.reset()
        states = torch.zeros((env.depth, env.depth, env.depth), device=env.device )
        next_states = torch.zeros((env.depth, env.depth, env.depth), device=env.device) 
        rewards = torch.zeros(env.depth, device=env.device)
        dones = torch.zeros(env.depth, device=env.device) 
        actions = torch.zeros(env.depth, dtype=torch.int64, device=env.device) 
        true_rewards = torch.zeros(env.depth, device=env.device)
        dists = torch.zeros((env.depth, 2), device=env.device ) 
        if reward_estimation:
            estimated_rewards = torch.zeros((env.depth,n_heads), device=env.device) 
        uncertainties_ep = torch.zeros(env.depth, device=env.device)
        shared_body_images = [] 

        # logging
        if episode % logging_interval == 0:
                print(f"[Progress] {int((episode/n_episode)*100)}%",flush=True)
        training_data["taus"].append(tau)
        if LOGGING and (episode % logging_interval == 0 or episode % periodic_logging == 0):
            if reward_estimation: 
                matched = reward_ensemble_to_tensor(env,uncertainty_scale,reward_ensemble)
            else:
                matched = seen_pairs[:, :,1 ]  + seen_pairs[:, :, 0 ] - 2
            snapshot_k_true,snapshot_k_false = critic_k_shared_to_tensor(env,shared_body,critic_k_head)
            if episode % periodic_logging == 0:
                training_data["periodic snapshots"].append( (critic_j_shared_to_tensor(env,shared_body,critic_j_head), 
                                                            snapshot_k_true, snapshot_k_false, 
                                                            policy_shared_to_tensor(env,shared_body,policy_head), 
                                                            matched.transpose(0, 1), 
                                                            episode))
            if episode % logging_interval == 0:
                if (saving_path is not None) :
                    utils.save_data(training_data,saving_path)
                training_data["exceptional snapshots"].append( (critic_j_shared_to_tensor(env,shared_body,critic_j_head), 
                                                                snapshot_k_true, snapshot_k_false, 
                                                                policy_shared_to_tensor(env,shared_body,policy_head), 
                                                                matched.transpose(0, 1), 
                                                                episode))
        # time measurement
        start_time_local = time.time()

        # play an episode using the current policy
        for step in range(env.depth):
            state_pair = env.state
            shared_body_images.append(shared_body(state.unsqueeze(0)).squeeze(0))
            dist_tensor = policy_head(shared_body_images[step].unsqueeze(0)).squeeze(0)
            action = torch.multinomial(dist_tensor,num_samples=1).squeeze(0)
            next_state, reward, done = env.step(int(action))

            if reward_estimation:        
                estimated_reward = torch.stack(reward_ensemble(state.unsqueeze(0)))
                estimated_reward = estimated_reward[:,0,action]
                with torch.no_grad():
                    uncertainty = uncertainty_scale * (torch.std(estimated_reward)** (pow_seen_pairs) )
                    uncertainty = torch.max(uncertainty,torch.tensor(0.00001))
            else:
                uncertainty = uncertainty_scale/ (seen_pairs[state_pair[0],state_pair[1],action] ** pow_seen_pairs)
            seen_pairs[state_pair[0],state_pair[1],action] += 1

            if step == 1 and action == 1-env.actions[step] and LOGGING:

                if len(training_data["uncertainty decrease"]) == 0:
                    print("frist uncertainty :",uncertainty)
                training_data["uncertainty decrease"].append( 1 / (uncertainty) )

            
            if reward_estimation:
                estimated_rewards[step] = estimated_reward
            states[step] = state
            next_states[step] = next_state
            rewards[step] = reward + uncertainty/(2*tau)
            true_rewards[step] = reward
            actions[step] = action
            dones[step] = done
            dists[step] = dist_tensor
            uncertainties_ep[step] = uncertainty

            # update state
            state = next_state
            # logging
            training_data["uncertainties"].append(uncertainty.item())


        cumulative_reward = int(torch.sum(true_rewards).item())
        # time measurement
        training_data["episode_time"] += time.time() - start_time_local
        start_time_local = time.time()

        # shared representation
        shared_body_images = torch.stack(shared_body_images)
        # H(pi)
        dists_cat = Categorical(probs=dists)
        entropies = dists_cat.entropy()
        entropies_detached = entropies.detach()
        # J(s)
        current_j = critic_j_head(shared_body_images)
        current_j_detached = current_j.detach()
        # K(s,a)
        q_states = critic_k_head(shared_body_images)
        current_q_states = q_states.gather(1,actions.unsqueeze(1))
        current_q_states_detached = current_q_states.detach()
        # Advantage
        with torch.no_grad():
            advantage_detached = current_q_states_detached - current_j_detached
        # Compute policy score 
        policy_score = - torch.mean( dists_cat.log_prob(actions) * (advantage_detached.squeeze(1)) + tau * entropies )
        # Compute critic_j loss
        with torch.no_grad():
            estimated_j_s = current_q_states_detached.squeeze(1) - tau * dists_cat.log_prob(actions)
        critic_j_loss = criterion(current_j.squeeze(1), estimated_j_s)
        # Compute critic_k loss using TD-lambda update
        cumul_sum_rewards = torch.zeros(env.depth+1, device=env.device)
        cumul_sum_entropies =torch.zeros(env.depth+1, device=env.device)
        for i in range(env.depth):
            cumul_sum_rewards[i+1] = cumul_sum_rewards[i] + rewards[i]
            cumul_sum_entropies[i+1] = cumul_sum_entropies[i] + entropies_detached[i]
        critic_k_loss = 0
        for start in range(env.depth):
            G = 0
            lambda_pow = 1
            for stop in range(start+1, env.depth):
                # G_n = r_start + (r_(start+1) + H(start+1)) + ... + (r_(stop-1)+H(stop-1)) + J(s_stop)
                G_n = (cumul_sum_rewards[stop] - cumul_sum_rewards[start]) + tau * (cumul_sum_entropies[stop] - cumul_sum_entropies[start+1]) +  current_j_detached[stop].item()
                G += lambda_pow * G_n
                lambda_pow *= Lambda
            G = (1 - Lambda)*G + lambda_pow*(cumul_sum_rewards[env.depth]-cumul_sum_rewards[start])
            critic_k_loss = critic_k_loss + criterion(current_q_states.squeeze(1)[start],G) / env.depth
        # Compute reward loss
        if reward_estimation:
            true_rewards = true_rewards.unsqueeze(1).repeat(1,n_heads)
            reward_loss = nn.MSELoss(reduction="mean")(estimated_rewards, true_rewards)

        # time measurement
        training_data["compute_loss_time"] += time.time() - start_time_local
        start_time_local = time.time()

         # update tau
        with torch.no_grad():
            prev_tau = tau
            inv_tau_sq = 1.0 / (2.0 * tau * tau)
            delta = entropies_detached - uncertainties_ep * inv_tau_sq
            tau_update = torch.mean(delta)
            tau = tau - learning_rate * tau_update
            tau = torch.clamp(tau, min=min_tau, max=prev_tau + step_bound_tau)
        # update critics and policy
        loss_shared = policy_score + critic_j_loss + critic_k_loss
        optimizer_shared.zero_grad()
        loss_shared.backward()
        if gradient_clipping != None:
            torch.nn.utils.clip_grad_norm_(
                list(shared_body.parameters()) +
                list(policy_head.parameters()) +
                list(critic_j_head.parameters()) +
                list(critic_k_head.parameters()),
                max_norm=gradient_clipping
            )
        optimizer_shared.step()
        # update reward ensemble
        if reward_estimation:
            optimizer_reward_ensemble.zero_grad()
            reward_loss.backward()
            if gradient_clipping_reward_ensemble != None:
                torch.nn.utils.clip_grad_norm_(
                    reward_ensemble.parameters(),
                    max_norm=gradient_clipping_reward_ensemble
                )
            optimizer_reward_ensemble.step()
        
        # logging
        training_data["update_time"] += time.time() - start_time_local
        if cumulative_reward == 1:
            training_data["successful episodes"].append(episode)
        training_data["cumulative_rewards"].append(cumulative_reward)
        training_data["actions history"].append(actions)
        training_data["episode cumulative uncertainty"].append(uncertainties_ep.mean().item())
        training_data["episode cumulative entropies"].append(entropies.mean().item())
        success_count += cumulative_reward
        if cumulative_reward == 1 and training_data["episodes to reach first reward"] is None:
            training_data["episodes to reach first reward"] = episode
            print("[Progress] episodes to reach first reward:", training_data["episodes to reach first reward"],flush=True)
            if early_stopping_first_reward:
                break
        if success_count/(episode+1) >= 0.1:
            training_data["episodes to reach 10% success"] = episode
            break

    if (saving_path is not None) :
        utils.save_data(training_data,saving_path)

    training_data["total_compute_time"] = str(timedelta(seconds=int(time.time()-start_time_global)))
    training_data["compute_loss_time"] = str(timedelta(seconds=int(training_data["compute_loss_time"])))
    training_data["update_time"] = str(timedelta(seconds=int(training_data["update_time"])))
    training_data["episode_time"] = str(timedelta(seconds=int(training_data["episode_time"])))

    return training_data

def run_ersac(noise_probability,seed,depth,learning_rate,gradient_clipping, init_tau,min_tau, pow_seen_pairs,Lambda, uncertainty_scale, n_episode,mid_size,reward_estimation,n_heads,saving_path):
    """
    Runs the ERSAC algorithm on the DeepSea environment with specified hyperparameters.

    This function initializes the environment and the neural network components used 
    in ERSAC, compiles them with `torch.compile`, and launches the training loop. 
    It returns the training data and metadata (e.g., hyperparameters).

    Parameters
    ----------
    noise_probability : float
        Probability to take the opposite action in the DeepSea environment. The basic DeepSea environment has a 0% chance to take the opposite action.
    seed : int
        Random seed for reproducibility.
    depth : int
        Depth (i.e., number of steps) of the DeepSea environment.
    learning_rate : float
        Learning rate for the optimizer.
    gradient_clipping : float
        Maximum gradient norm for clipping.
    init_tau : float
        Initial risk parameter.
    min_tau : float
        Minimum risk parameter
    pow_seen_pairs : float
        Power used to scale exploration bonus based on the number of (state, action) visits.
    Lambda : float
        Used in the TD(Lambda) update to learn K.
    uncertainty_scale : float
        Scale factor for the epistemic uncertainty bonus.
    n_episode : int
        Maximum number of episodes to train.
    mid_size : int
        Width of the shared hidden layer in the networks.
    reward_estimation : bool
        Whether the uncertainty is modeled by explicit counting or by reward ensemble.
    n_heads : int
        Number of ensemble heads used for reward uncertainty estimation.
    saving_path : str
        Path to save the training data and optionally checkpoints or logs.

    Returns
    -------
    training_data : dict
        Dictionary containing performance metrics, configuration, and other training metadata.
    """
    
    utils.set_seed(seed)
    env = DeepSea(depth,noise_probability=noise_probability)

    reward_ensemble = Reward(env,n_heads=n_heads,n_mid=mid_size)
    shared_body = Shared_body(env,mid_size)
    policy_head = Policy_head(env,mid_size)
    critic_j_head = Critic_j_head(env,mid_size)
    critic_k_head = Critic_k_head(env,mid_size)

    policy_head = policy_head.to(env.device)
    critic_j_head = critic_j_head.to(env.device)
    critic_k_head = critic_k_head.to(env.device)
    shared_body = shared_body.to(env.device)
    reward_ensemble = reward_ensemble.to(env.device)

    # Then compile the modules
    policy_head = torch.compile(policy_head)
    critic_j_head = torch.compile(critic_j_head)
    critic_k_head = torch.compile(critic_k_head)
    shared_body = torch.compile(shared_body)
    reward_ensemble = torch.compile(reward_ensemble)

    training_data = ersac(
                env=env,seed=seed,shared_body=shared_body,policy_head=policy_head,critic_j_head=critic_j_head,critic_k_head=critic_k_head,reward_ensemble=reward_ensemble,
                learning_rate=learning_rate, gradient_clipping=gradient_clipping,Lambda=Lambda, uncertainty_scale=uncertainty_scale,mid_size=mid_size, reward_estimation=reward_estimation, n_heads=n_heads,init_tau=init_tau,min_tau=min_tau,pow_seen_pairs=pow_seen_pairs,
                n_episode=n_episode,
                early_stopping_first_reward=False,
                saving_path=saving_path
                )
    training_data["hyperparameters"]["seed"] = seed
    return training_data