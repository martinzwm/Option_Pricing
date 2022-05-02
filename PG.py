import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from LSTM import data_gen, SimDataset
from tqdm import tqdm

# For now just copy paste the code from the Colab solution (PSet 6)
class PGNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim = 64, layer = 2):
        super(PGNetwork, self).__init__()
        if layer == 0:
            self.net = nn.Sequential(nn.Linear(in_dim, out_dim))

        else:
            modules = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            for i in range(layer - 1):
                modules.append(nn.Linear(hidden_dim, hidden_dim))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(hidden_dim, out_dim))
            self.net = nn.Sequential(*modules)

    def forward(self, observations):
        B, N = observations.size()
        # N should really be 1. 
        self.net(observations)
        return Categorical(logits = F.log_softmax(x, dim = 1))

class ValueNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim = 64):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, observation):
        return self.net(observation)

# Next, compute the discounted returns, same as how we did in PSet 6. 
def discounted_returns(rewards, gamma, device = None):
    returns = torch.zeros_like(rewards, device = device)
    returns[-1] = rewards[-1]
    for i, r in enumerate(reversed(rewards[:, -1])):
        # TODO: vectorize this. 
        returns[-i-2] = r + gamma * returns[-i-1]

def update_parameters(optimizer, model, rewards, log_probs, gamma,
                      values = None, temporal = False, device = None):
    policy_loss = []
    returns = discounted_returns(rewards, gamma, device)
    eps = np.finfo(np.float32).eps.item()
    discount = torch.zeros_like(rewards, device = device)
    discount[0] = 1
    for i in range(1, len(discount)):
        discount[i] = gamma * discount[i - 1]

    if values != None:
        value_loss = F.smooth_l1_loss(values, returns)
        returns -= values.detach()

    policy_loss = - torch.sum(log_probs * returns * discount) if temporal else - torch.sum(log_probs) * returns[0]
    if values != None:
        loss = policy_loss + value_loss
    else:
        loss = policy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def reward_func(traj, strike):
    ans = torch.max(0, traj - strike) # or is it the other way round? 
    return ans[, 1:] - ans[, :-1]

def rollout(model, traj, dt, vmodel=None, device=None, MAX_T=10000):
    actions = torch.zeros(MAX_T, device = device, dtype=torch.int)
    rewards = torch.zeros(MAX_T, device=device)
    log_probs = torch.zeros(MAX_T, device=device)
    values = torch.zeros(MAX_T, device=device)
    ep_reward = 0
    for T in range(MAX_T):
        dt_vec = torch.repeat((MAX_T - T) * dt, batch_size)
        state = torch.cat((traj[:, T], dt_vec))
        dist = model(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if vmodel:
            value = vmodel(x)
            values[T] = value
        done = (action == 1 or T == MAX_T - 1)
        if not done:
            # Chooses not to exercise. 
            reward = reward_func(traj)[T] # TODO: change this based on the true value function!
        else:
            reward = max(traj[T] - strike, 0)
            ep_reward = reward
        rewards[T] = reward
        actions[T] = action
        log_probs[T] = log_prob
        obs = next_obs
        if action == 1:
            break
    return actions[:T + 1], rewards[:T + 1], log_probs[:T + 1], values[:T + 1], ep_reward

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device('cuda'):
        batch_size = 64
    else:
        batch_size = 8
    X, S0, r, sigma, T, M, N, transition = 40, 36, 0.06, 0.2, 1, 100, 800, BrownianMotion
    data = data_gen(X, S0, r, sigma, T, M, N, transition)
    # TODO: data_load
    train_set = SimDataset(train_set)
    test_set = SimDataset(test_set)

    train_loader = DataLoader(
        train_set,
        batch_size = batch_size,
        num_workers 1,
        shuffle = True
    )
    test_loader = DataLoader(
        test_set,
        batch_size = batch_size,
        num_workers = 1,
        shuffle = True
    )
    num_epochs = 100 # Change later. 
    for step in range(num_epochs):
        for item in train_loader:
            actions, rewards, log_probs, valyes, ep_reward = rollout(
                model, item, vmodel=vmodel if USE_VALUE_NETWORK else None, device=device)

            update_parameters(optimizer, model, rewards, log_probs, gamma, 
                              values = values if USE_VALUE_NETWORK else None, 
                              temporal = TEMPORAL, device = device)
