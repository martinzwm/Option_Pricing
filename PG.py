import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import argparse
import pandas as pd
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
from torch.optim import *
from utility import SimDataset, load_checkpoint, save_checkpoint
from transition import BrownianMotion
from LSM import AmericanOptionsLSMC

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
        B, L = observation.size()
        return self.net(observation).view(B)

# Next, compute the discounted returns, same as how we did in PSet 6. 
def discounted_returns(rewards, gamma, device = None):
    """
        Args:
            rewards: B x L, B = # trajectory, L = length of each trajectory. 
            gamma: float
    """
    returns = torch.zeros_like(rewards, device = device)
    returns[:, -1] = rewards[:, -1]
    for i in range(rewards.shape[1] - 1, 0, -1):
        r = rewards[:, i - 1]
        returns[:, i - 1] = r + gamma * returns[:, i]
    return returns

def get_Q(price, payoff, gamma):
    """
        Args:
            price: B x L
            payoff: B x L, B = num_batches
        Returns:
            Q: B x L
    """
    B, L = payoff.shape
    q_matrix = np.zeros_like(payoff)
    q_matrix[:, -1] = payoff[:, -1]
    for t in range(L - 2, -1, -1):
        # In the case where price[:, t] are all the same, we just take the mean. 
        # This would happen in the beginning of the trajectory. 
        if np.var(price[:, t]) < 1e-9:
            cont_val = np.mean(q_matrix[:, t + 1]) * gamma
        else:
            regress = np.polyfit(price[:, t], q_matrix[:, t + 1] * gamma, 5)
            cont_val = np.polyval(regress, price[:, t])
        q_matrix[:, t] = np.where(payoff[:, t] > cont_val, payoff[:, t], 
                                  q_matrix[:, t + 1] * gamma)
    return q_matrix

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

# Note: don't use TD(0) yet. Might be difficult because of the gamma factor. 
def reward_func(traj, strike, style = None):
    assert(style in [None, "TD0"])
    if style is None:
        reward = torch.clamp(strike - traj, min = 0)
    else:
        ans = torch.clamp(strike - traj, min=0)
        reward = ans[:, 1:] - ans[:, :-1]
    # TODO: change this into some sort of TD(0) algo. 
    return reward

# Here, we want to do one step of simulation. 
# For train, we have the model to act probabilistically. 
# For test, we compute the _expected_ reward. 
def rollout(model, traj, strike, dt, gamma, 
            vmodel=None, mode = "train", device=None):
    """
        Args:
            model: the PG model (presumably, no other choice)
            traj: B x L, B = # number of trajectory (a.k.a. batch size), L = trajectory length
            strike: the strike price
            dt: the difference in time (T / M)
            gamma: discount factor. 
            vmodel: value network, or simply, "actor critic". 
            mode: train, or test
            device
    """
    assert mode in ["train", "test"], "only train / test mode makes sense here. "
    B, L, _ = traj.size()
    actions = torch.zeros((B, L), device = device, dtype=torch.int)
    rewards = torch.zeros((B, L), device = device)
    log_probs = torch.zeros((B, L), device = device)
    values = torch.zeros((B, L), device = device)
    ep_reward = torch.zeros((B), device = device)
    # For train, the exercised parameter will just be deterministic (0 or 1). 
    # For test, this is the probability that we have not exercised yet. 
    exercised = torch.zeros((B), device = device)
    reward_style = None
    reward_list = reward_func(traj[:, :, 0], strike, reward_style)
    if vmodel == "actor_critic":
        q_all = get_Q(traj[:, :, 0].cpu().numpy(), reward_list.cpu().numpy(), gamma)
        q_all = torch.from_numpy(q_all).float().to(device)
        q_state = q_all[:, 0]
    
    if reward_style == "TD0":
        base_val = max(strike - traj[0, 0, 0].item(), 0)
        ep_reward.fill_(base_val)
        # ep_reward = torch.full((B), base_val)

    for T in range(L):
        state = traj[:, T].float() # B x 2
        dist = model(state)
        if mode == "train": 
            action = dist.sample() # Length B
            log_prob = dist.log_prob(action) # Length B
            # If we've already exercised then the action no longer matters. 
            action = torch.logical_and(action, torch.logical_not(exercised))
            exercised = torch.logical_or(action, exercised)
            if vmodel == "actor_critic": # We'll fix this later.
                if T < L - 1:
                    q_state = torch.where(exercised, q_state, q_all[:, T + 1])
                values[:, T] = q_state

            elif vmodel is not None:
                st_act = torch.cat((state, action.view(B, 1)), axis = -1)
                values[:, T] = vmodel(st_act)
            
            if reward_style is None:
                if T < L - 1:
                    reward = reward_list[:, T] * action.float()
                else:
                    count = torch.logical_or(action, torch.logical_not(exercised))
                    reward = reward_list[:, T] * count.float()
                rewards[:, T] = reward
                ep_reward += reward

            else:
                # TD(0) case. 
                if T < L - 1:
                    reward = (1 - exercised.float()) * reward_list[:, T]
                    rewards[:, T] = reward
                    ep_reward += reward
            actions[:, T] = action
            log_probs[:, T] = log_prob
        else:
            # Test mode. 
            ones = torch.ones((B), device = device)
            cond_prob = torch.exp(dist.log_prob(ones))
            real_prob = cond_prob * (1 - exercised)
            ep_reward += reward_list[:, T] * real_prob * (gamma ** T)
            exercised += real_prob
            if T == L - 1:
                # Last minute of non-exercising. 
                ep_reward += reward_list[:, T] * (1 - exercised) * (gamma ** T)

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

class PGRunner():
    def __init__(self, config):
        self.config = config
        self.strike = config.strike
        self.S0 = config.S0
        self.sigma = config.sigma
        self.r = config.r
        self.T = config.T
        self.M = config.M
        self.N = config.N
        self.transition = BrownianMotion
        self.dt = self.T / self.M
        self.gamma = np.exp(-self.r * self.dt)
        self.temporal = config.temporal
        self.vmodel = config.vmodel if "vmodel" in config else None
        self.prob_fname = config.prob_fname
        self.data = self.gen_data()
        self.train_loader, self.test_loader = self.get_loader()


    def gen_data(self):
        transition_model = BrownianMotion(self.S0, self.r, self.sigma, self.T, self.M, self.N)
        data = np.transpose(transition_model.simulate())
        return data

    def get_loader(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == torch.device('cuda'):
            batch_size = 20
        else:
            batch_size = 20
        train_set, test_set = data_load_pg(self.data, 0.80)
        train_set = SimDataset(train_set)
        test_set = SimDataset(test_set)

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
        return train_loader, test_loader

    # The previous test loader during training is more like validation. 
    # But here, we want to check if it's best. 
    def gen_loader_test(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        N_test = 100000
        transition_model = BrownianMotion(self.S0, self.r, self.sigma, self.T, self.M, N_test)
        data = np.transpose(transition_model.simulate(seed = 2077)) 
        if device == torch.device('cuda'):
            batch_size = 50
        else:
            batch_size = 200
        train_set, test_set = data_load_pg(data, 0.00)
        test_set = SimDataset(test_set)
        test_loader = DataLoader(
            test_set,
            batch_size = batch_size,
            num_workers = 1,
            shuffle = True
        )
        return test_loader



    def train(self):
        config = self.config
        num_epochs = config.num_epochs
        LR = float(config.LR)
        torch.manual_seed(123)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PGNetwork(in_dim = 2, out_dim = 2).to(device)

        if self.vmodel == "vmodel":
            vmodel = ValueNetwork(3, 64).to(device)
            optimizer = optim.Adam(list(model.parameters()) + list(vmodel.parameters()), lr=LR)
        else:
            vmodel = self.vmodel # None, or actor critic
            optimizer = optim.Adam(model.parameters(), lr=LR)

        
        for step in range(num_epochs):
            model.train()
            for item in tqdm(self.train_loader):
                item = item.to(device)
                actions, rewards, log_probs, values, ep_reward = rollout(
                    model, item, self.strike, self.dt, self.gamma, 
                    vmodel=vmodel, mode = "train", device=device)
                # print(torch.mean(ep_reward))

                update_parameters(optimizer, model, rewards, log_probs, self.gamma, 
                                  values = values if vmodel is not None else None, 
                                  temporal = self.temporal, device = device)
            
            test_reward = []
            model.eval()
            save_checkpoint(model, optimizer, step, config.save_dir)
            # Check every 10 epochs. 
            if (step + 1) % 10 == 0:
                for item in tqdm(self.test_loader):
                    item = item.to(device)
                    actions, rewards, log_probs, values, ep_reward = rollout(
                        model, item, self.strike, self.dt, self.gamma, 
                        vmodel=vmodel, mode = "test", device=device)
                    for rew in ep_reward:
                        test_reward.append(rew.item())
                print(sum(test_reward)/len(test_reward))
                prices = np.linspace(np.min(self.data), np.max(self.data), 40)
                dtimes = np.linspace(0, 1, 51)
                # prob_plot(model, prices, dtimes = dtimes, fname = self.prob_fname) 
            print("Epoch {} done".format(step + 1))

    def test(self):
        config = self.config
        torch.manual_seed(123)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PGNetwork(in_dim = 2, out_dim = 2).to(device)

        if self.vmodel == "vmodel":
            vmodel = ValueNetwork(3, 64).to(device)
        else:
            vmodel = self.vmodel # None, or actor critic
        save_step = config.save_step if "save_step" in config else "max"
        load_dir = config.load_dir
        load_checkpoint(model, optimizer = None, step = save_step, save_dir = load_dir)
        test_reward = []
        model.eval()
        new_test_loader = self.gen_loader_test()
        for item in tqdm(new_test_loader):
            item = item.to(device)
            actions, rewards, log_probs, values, ep_reward = rollout(
                model, item, self.strike, self.dt, self.gamma,
                vmodel=vmodel, mode = "test", device=device)
            for rew in ep_reward:
                test_reward.append(rew.item())
        test_reward = np.array(test_reward)
        import scipy.stats as st
        np.save("returns.npy", test_reward)
        print(np.mean(test_reward))
        print(st.t.interval(0.95, test_reward.shape[0] - 1, 
                            loc=np.mean(test_reward), scale=st.sem(test_reward)))
        dtime_min = int(np.min(self.data))
        dtime_max = int(np.max(self.data))+ 1
        prices = np.arange(dtime_min, dtime_max)
        dtimes = np.linspace(0, 1, 51)
        prob_plot(model, prices, dtimes = dtimes, end_time = self.T, fname = self.prob_fname, 
                 csv_fname = config.csv_fname)
        

def prob_plot(model, prices, dtimes, end_time, fname = None, csv_fname = None):
    """
        Args: 
            model: PGNet (only one choice)
            prices: a list of all the prices
            dtimes: a fixed set of time intervals
            end_time: maturity time
        Returns:
            a plot
    """
    # First need to process first. 
    model.eval()
    L = prices.shape[0]
    T = dtimes.shape[0]
    probs = np.empty((T, L))
    for (i, dtime) in enumerate(dtimes): 
        dtime_vec = np.repeat(dtime, L, 0)
        state = np.stack([prices, dtime_vec], axis = 1)
        state_tensor = torch.from_numpy(state).float()
        dist = model(state_tensor)
        exercise = torch.tensor([1] * L)
        pr = torch.exp(dist.log_prob(exercise)).cpu().detach().numpy()
        probs[i] = pr
    if fname == None:
        fname = 'pg_prob.png'
    if csv_fname == "None":
        csv_fname = 'pg_prob.csv'
    df = pd.DataFrame(probs, columns = prices)
    df['Time'] = 1.00 - dtimes
    df.to_csv(csv_fname)
    
    X, Y = np.meshgrid(prices, 1.00 - dtimes) # Here, we need to
    plt.figure()
    plt.contourf(X, Y, probs)
    plt.xlabel("Stock")
    plt.ylabel("Time")
    plt.set_cmap('jet');
    plt.colorbar()
    plt.savefig(fname)
    plt.clf()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process config file.')
    parser.add_argument('config_file')
    args = parser.parse_args()
    config_file = args.config_file
    config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))
    runner = PGRunner(config)
    runner.test() # Feel free to change to train, if needed. 
