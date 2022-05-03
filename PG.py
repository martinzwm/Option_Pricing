import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from transition import BrownianMotion
from torch.utils.data import Dataset, DataLoader
from torch.optim import *
from LSTM import data_gen, SimDataset
from tqdm import tqdm

# For now just copy paste the code from the Colab solution (PSet 6)
class PGNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim = 64, num_layer = 2):
        super(PGNetwork, self).__init__()
        if num_layer == 0:
            self.net = nn.Sequential(nn.Linear(in_dim, out_dim))

        else:
            modules = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            for i in range(num_layer - 1):
                modules.append(nn.Linear(hidden_dim, hidden_dim))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(hidden_dim, out_dim))
            self.net = nn.Sequential(*modules)
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, observations):
        B, N = observations.size()
        # B should really be 1. 
        x = self.net(observations)
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
    returns[:, -1] = rewards[:, -1]
    for i in range(rewards.shape[1] - 1, 0, -1):
        r = rewards[:, i - 1]
        returns[:, i-2] = r + gamma * returns[:, i-1]
    return returns

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
    if temporal:
        policy_loss = - torch.sum(log_probs * returns * discount)
    else: 
        policy_loss = -torch.sum(torch.sum(torch.sum(log_probs, dim = 1) * returns[:, 0]))
    if values != None:
        loss = policy_loss + value_loss
    else:
        loss = policy_loss
    # loss = Variable(loss, requires_grad = True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def reward_func(traj, strike):
    ans = torch.clamp(strike - traj, min=0)
    # TODO: change this into some sort of TD(0) algo. 
    return ans[:, 1:] - ans[:, :-1]

def rollout(model, traj, strike, dt, vmodel=None, device=None):
    """
        Args:
            model: the PG model (presumably, no other choice)
            traj: B x L, B = # number of trajectory (a.k.a. batch size), L = trajectory length
    """
    # from IPython import embed; embed()
    B, L = traj.size()
    # assert(B == 1), "batch size of > 1 not supported yet."
    actions = torch.zeros((B, L), device = device, dtype=torch.int)
    rewards = torch.zeros((B, L), device=device)
    log_probs = torch.zeros((B, L), device=device)
    values = torch.zeros((B, L), device=device)
    ep_reward = 0
    exercised = torch.zeros((B), device = device)
    reward_list = reward_func(traj, strike)
    for T in range(L):
        dt_vec = torch.full(torch.Size([B]), (L - T) * dt)
        state = torch.stack([traj[:, T], dt_vec], dim = 1).float()
        dist = model(state)
        action = dist.sample() # Length B
        log_prob = dist.log_prob(action) # Length B
        # If we've already exercised then the action no longer matters. 
        action = torch.logical_and(action, torch.logical_not(exercised))
        exercised = torch.logical_or(action, exercised)
        if vmodel: # We'll fix this later. , TODO
            value = vmodel(x)
            values[:, T] = value
        
        #ep_reward = reward
        if T < L - 1:
            reward = (1 - exercised.float()) * reward_list[:, T]
            rewards[:, T] = reward
        actions[:, T] = action
        log_probs[:, T] = log_prob
    #print(actions)
    #print(rewards)
    return actions, rewards, log_probs, values, ep_reward

def data_load_pg(traj, train_test_split=0.8):
    """ 
        Args: 
            data: trajcetory
    """
    train_size = int(np.round(train_test_split * traj.shape[0]))

    train_set, test_set = traj[:train_size], traj[train_size:]

    return train_set, test_set

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device('cuda'):
        batch_size = 10
    else:
        batch_size = 10
    X, S0, r, sigma, T, M, N, transition = 40, 36, 0.06, 0.2, 1, 10, 100, BrownianMotion
    strike = 40
    transition_model = BrownianMotion(S0, r, sigma, T, M, N)
    data = transition_model.simulate()
    # TODO: data_load
    train_set, test_set = data_load_pg(data, 0.80)
    train_set = SimDataset(train_set)
    test_set = SimDataset(test_set)
    dt = T / M

    train_loader = DataLoader(
        train_set,
        batch_size = batch_size,
        num_workers = 1,
        shuffle = True
    )
    test_loader = DataLoader(
        test_set,
        batch_size = batch_size,
        num_workers = 1,
        shuffle = True
    )
    LR = 1e-3
    USE_VALUE_NETWORK = False
    TEMPORAL = False
    model = PGNetwork(in_dim = 2, out_dim = 2)
    gamma = np.exp(-r * dt)
    if USE_VALUE_NETWORK:
        vmodel = ValueNetwork(env.observation_space.shape[0], 64).to(device)
        optimizer = optim.Adam(list(model.parameters()) + list(vmodel.parameters()), lr=LR)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR)
    num_epochs = 100 # Change later. 
    for step in tqdm(range(num_epochs)):
        for item in train_loader:
            actions, rewards, log_probs, values, ep_reward = rollout(
                model, item, strike, dt, vmodel=vmodel if USE_VALUE_NETWORK else None, device=device)

            update_parameters(optimizer, model, rewards, log_probs, gamma, 
                              values = values if USE_VALUE_NETWORK else None, 
                              temporal = TEMPORAL, device = device)
            # from IPython import embed; embed()

if __name__ == "__main__":
    train()
