import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
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

    def forward(self, observation):
        self.net(observation)
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
