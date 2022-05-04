import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from transition import BrownianMotion
from torch.utils.data import Dataset, DataLoader
from torch.optim import *
from utility import SimDataset
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
                if i % 2 == 0:
                    modules.append(nn.BatchNorm1d(hidden_dim))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(hidden_dim, out_dim))
            self.net = nn.Sequential(*modules)
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, observations):
        B, N = observations.size()
        # N should really be 2, one for time and . 
        try:
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
        # TODO: add the Actor-Critic version (pass in the feature vector). 

    def forward(self, observation):
        B, L = observation.size()
        return self.net(observation).view(B)

# Next, compute the discounted returns, same as how we did in PSet 6. 
def discounted_returns(rewards, gamma, device = None):
    returns = torch.zeros_like(rewards, device = device)
    returns[:, -1] = rewards[:, -1]
    for i in range(rewards.shape[1] - 1, 0, -1):
        r = rewards[:, i - 1]
        returns[:, i-1] = r + gamma * returns[:, i]
    return returns

def update_parameters(optimizer, model, rewards, log_probs, gamma,
                      values = None, temporal = False, device = None):
    model.train()
    optimizer.zero_grad()
    policy_loss = []
    returns = discounted_returns(rewards, gamma, device)
    eps = np.finfo(np.float32).eps.item()
    discount = torch.zeros_like(rewards, device = device)
    discount[:, 0] = 1
    for i in range(1, discount.shape[1]):
        discount[:, i] = gamma * discount[:, i - 1]

    if values != None:
        value_loss = F.smooth_l1_loss(values, returns)
        returns = returns - values.detach()
    if temporal:
        policy_loss = - torch.sum(log_probs * returns * discount)
    else: 
        policy_loss = -torch.sum(torch.sum(torch.sum(log_probs, dim = 1) * returns[:, 0]))
    if values != None:
        loss = policy_loss + value_loss
    else:
        loss = policy_loss
    # loss = Variable(loss, requires_grad = True)
    loss.backward()
    optimizer.step()

def reward_func(traj, strike, style = None):
    assert(style in [None, "TD0"])
    if style is None:
        reward = torch.clamp(strike - traj, min = 0)
    else:
        ans = torch.clamp(strike - traj, min=0)
        reward = ans[:, 1:] - ans[:, :-1]
    # TODO: change this into some sort of TD(0) algo. 
    return reward

def rollout(model, traj, strike, dt, vmodel=None, device=None):
    """
        Args:
            model: the PG model (presumably, no other choice)
            traj: B x L, B = # number of trajectory (a.k.a. batch size), L = trajectory length
    """
    B, L, _ = traj.size()
    actions = torch.zeros((B, L), device = device, dtype=torch.int)
    rewards = torch.zeros((B, L), device = device)
    log_probs = torch.zeros((B, L), device = device)
    values = torch.zeros((B, L), device = device)
    ep_reward = torch.zeros((B), device = device)
    exercised = torch.zeros((B), device = device)
    reward_style = None
    reward_list = reward_func(traj[:, :, 0], strike, reward_style)
    if reward_style == "TD0":
        base_val = max(strike - traj[0, 0, 0].item(), 0)
        ep_reward.fill_(base_val)
        # ep_reward = torch.full((B), base_val)
    for T in range(L):
        state = traj[:, T].float() # B x 2
        dist = model(state)
        action = dist.sample() # Length B
        log_prob = dist.log_prob(action) # Length B
        # If we've already exercised then the action no longer matters. 
        action = torch.logical_and(action, torch.logical_not(exercised))
        exercised = torch.logical_or(action, exercised)
        if vmodel: # We'll fix this later.
            value = vmodel(state)
            values[:, T] = value
        
        if reward_style is None:
            if T < L - 1:
                reward = reward_list[:, T] * action.float()
            else:
                count = torch.logical_or(action, torch.logical_not(exercised))
                reward = reward_list[:, T] * count.float()
            rewards[:, T] = reward
            ep_reward += reward
        else:
            if T < L - 1:
                reward = (1 - exercised.float()) * reward_list[:, T]
                rewards[:, T] = reward
                ep_reward += reward
        actions[:, T] = action
        log_probs[:, T] = log_prob
    #print(actions)
    #print(rewards)
    return actions, rewards, log_probs, values, ep_reward

def data_load_pg(traj, train_test_split=0.8):
    """ 
        Args: 
            data: trajcetory, np array size N x L
            T: time horizon. 
        Returns:
            train_set: N1 x L x 2
            test_set: N2 x L x 2
    """
    # Idea is to secretly inject dt into the vector. 
    N, L = traj.shape
    dt_vec = np.flip(np.arange(L) / L, axis = 0)[np.newaxis, :]
    traj_full = np.stack([traj, np.repeat(dt_vec, N, 0)], axis = 2)
    train_size = int(np.round(train_test_split * traj.shape[0]))

    train_set, test_set = traj_full[:train_size], traj_full[train_size:]

    return train_set, test_set

def pg_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device('cuda'):
        batch_size = 20
    else:
        batch_size = 20
    strike, S0, r, sigma, T, M, N, transition = 40, 36, 0.06, 0.2, 1, 50, 1000, BrownianMotion
    transition_model = BrownianMotion(S0, r, sigma, T, M, N)
    data = np.transpose(transition_model.simulate())
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
        vmodel = ValueNetwork(2, 64).to(device)
        optimizer = optim.Adam(list(model.parameters()) + list(vmodel.parameters()), lr=LR)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR)
    num_epochs = 100 # Change later. 
    for step in tqdm(range(num_epochs)):
        model.train()
        for item in train_loader:
            actions, rewards, log_probs, values, ep_reward = rollout(
                model, item, strike, dt, vmodel=vmodel if USE_VALUE_NETWORK else None, device=device)
            # print(torch.mean(ep_reward))

            update_parameters(optimizer, model, rewards, log_probs, gamma, 
                              values = values if USE_VALUE_NETWORK else None, 
                              temporal = TEMPORAL, device = device)
        
        test_reward = []
        model.eval()
        if (step + 1) % 10 == 0:
            for item in test_loader:
                actions, rewards, log_probs, values, ep_reward = rollout(
                    model, item, strike, dt, vmodel=vmodel if USE_VALUE_NETWORK else None, device=device)
                for rew in ep_reward:
                    test_reward.append(rew.item())
            print(sum(test_reward)/len(test_reward))
            prices = np.linspace(np.min(data), max(strike, S0), 40)
            dtimes = np.linspace(0, 1, 6)
            prob_plot(model, prices, dtimes = dtimes, fname = "pg_plot.png") 
        

def prob_plot(model, prices, dtimes, fname = None):
    """
        Args: 
            model: PGNet (only one choice)
            prices: a list of all the prices
            dtimes: a fixed set of time intervals
        Returns:
            a plot
    """
    # First need to process first. 
    model.eval()
    L = prices.shape[0]
    for dtime in dtimes: 
        dtime_vec = np.repeat(dtime, L, 0)
        state = np.stack([prices, dtime_vec], axis = 1)
        state_tensor = torch.from_numpy(state).float()
        dist = model(state_tensor)
        exercise = torch.tensor([1] * L)
        probs = torch.exp(dist.log_prob(exercise))
        probs = probs.cpu().detach().numpy()
        plt.plot(prices, probs, label = "dt = {:.2f}".format(dtime))
    if fname == None:
        fname = 'pg_prob.png'
    plt.legend()
    plt.savefig(fname)
    plt.clf()

if __name__ == "__main__":
    pg_train()
