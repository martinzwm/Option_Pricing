# File for Q-learning. 

# We consider the inspiration by https://github.com/wuxx1016/Reinforcement-Learning-in-Finance/blob/master/Week3/dp_qlbs_oneset_m3_ex3_v4.ipynb

import numpy as np
import pandas as pd
from scipy.stats import norm
import random

import sys

import time
import matplotlib.pyplot as plt
from tqdm import tqdm
# Here, the basis construction apparently uses bspline
import bspline
import bspline.splinelab as splinelab

# First, let's get the Black-Scholes prices. 
def bs_put(t, S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 1/2 * sigma**2) * (T-t)) / sigma / np.sqrt(T-t)
    d2 = (np.log(S0/K) + (r - 1/2 * sigma**2) * (T-t)) / sigma / np.sqrt(T-t)
    price = K * np.exp(-r * (T-t)) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return price

def bs_call(t, S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 1/2 * sigma**2) * (T-t)) / sigma / np.sqrt(T-t)
    d2 = (np.log(S0/K) + (r - 1/2 * sigma**2) * (T-t)) / sigma / np.sqrt(T-t)
    price = S0 * norm.cdf(d1) - K * np.exp(-r * (T-t)) * norm.cdf(d2)
    return price

def d1(S0, K, r, sigma, T):
    return (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))

def d2(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))

# Might as well implement DiscreteBlackScholes first. 
class DiscreteBS():
    def __init__(self, 
                 S0, 
                 strike, 
                 mu, 
                 sigma, 
                 r, 
                 T, 
                 num_steps, 
                 num_paths,
                 risk_lambda):
        self.S0 = S0
        self.strike = strike
        self.sigma = sigma #Volatility
        self.r = r # risk free rate
        self.mu = mu
        # self.M = M # number of states we wanna discretize
        # self.N = N # number of mesh grid in time domain
        self.T = T # time to marutiry, in years 
        self.num_steps = num_steps # number of time steps
        self.num_paths = num_paths # number of MC paths, also N_MC in some notation
        self.risk_lambda = risk_lambda # risk aversion. 
        self.stock_type = "American"
        self.option_type = "put"
        self.reg_param = 0.001

        self.dt = self.T / self.num_steps
        # Discount factor: the "annual" inflation is r and we consider each timestep as 1/num_steps
        self.gamma = np.exp(-r * self.dt)
        # Stock prices. 
        self.S = np.empty((self.num_paths, self.num_steps + 1))

        # Initialize half of the paths with stock prices (1+-0.5) s0
        # The other half start with s0
        half_paths = int(num_paths / 2)
        self.options = np.empty((self.num_paths, self.num_steps + 1))
        self.intrinsic = np.empty((self.num_paths, self.num_steps + 1))

        self.positions = np.zeros((self.num_paths, self.num_steps + 1)) # cash position
        self.opt_hedge = np.empty((self.num_paths, self.num_steps + 1)) # optimal hedge. 
        self.pi = np.empty((self.num_paths, self.num_steps + 1)) # Portfolio value. 
        self.pi_hat = np.empty((self.num_paths, self.num_steps + 1)) # Portfolio, de-meaned. 
        self.rewards = np.empty((self.num_paths, self.num_steps + 1)) # Reward function. 
        self.Q = np.empty((self.num_paths, self.num_steps + 1)) # Q function. 

        self.states = None
        self.features = None
        
        # Delta S is the temporal difference
        # Delta S hat is the de-meaned version. 
        self.delta_S = None
        self.delta_S_hat = None
        self.coef = 0 # or 1 / (2 * gamma * risk_lambda)
        self.update_states()

    # Get the temporal difference delta_S_hat and the states
    def update_states(self, seed = 42):
        self.S[:, 0] = self.S0 * np.ones(num_paths, 'float')
        np.random.seed(seed)
        #Z = np.random.normal(0, 1, size = (self.num_steps + 1, self.num_paths)).T
        Z = np.random.randn(self.num_paths, self.num_steps)
        #Z1 = np.random.normal(0, 1, size = (self.num_paths, self.num_steps + 1))

        # Here, the dynamics is 
        # S_{t+1} = S_t exp((mu-sigma**2/2)dt + sigma sqrt(dt) Z)
        # Repeat this for num_paths times. 
        for step in range(self.num_steps):
            exp_term = np.exp((self.mu - (self.sigma ** 2) / 2) * self.dt + (self.sigma * np.sqrt(self.dt) * Z[:, step]))
            self.S[:, step + 1] = self.S[:, step] * exp_term

        # Here, dS_t = S_{t+1} - exp(gamma) S_t
        self.delta_S = self.S[:, 1:] - np.exp(self.r * self.dt) * self.S[:, :self.num_steps]
        self.delta_S_hat = np.apply_along_axis(lambda x: x - np.mean(x), axis = 0, arr = self.delta_S)

        # Now, determine the states. 
        # This is given by X_t = -(mu - sigma**2/2) t dt + log(S_t)
        self.states = - (self.mu - self.sigma ** 2 / 2) * np.arange(self.num_steps + 1) * self.dt + np.log(self.S)
        
    # Here we generate our paths. 
    # To do this, we need to get the states and from those, get the basis functions using spline. 
    # This gives us the data matrices (data_mat in coursera, but here we just use self.feature). 
    # This can be changed to match how we actually generate the basis; for now leave it as it is to make sure the code matches the original repo. 
    def gen_basis(self, num_basis):
        # Get the feature vectors. 
        states_min = np.min(self.states)
        states_max = np.max(self.states)
        tau = np.linspace(states_min, states_max, num_basis)
        k = splinelab.aptknt(tau, 4) # order of spline = 4
        basis = bspline.Bspline(k, 4)
        self.features = np.zeros((self.num_steps + 1, self.num_paths, num_basis))
        for i in np.arange(self.num_steps + 1):
            x = self.states[:, i]
            self.features[i] = np.array([ basis(el) for el in x ])
        return self.features

    # Here, we start the code to calculate the optimal Q. 
    # In QLBS, the optimal action is given by a(X_t) = sum phi * Phi and Q given by sum omega * Phi
    # Where Phi is the feature vector (generated as basis). 
    # Now, Phi is A^{-1}B, A = sum Phi^k sum Phi^k.T dS**2, B = sum Phi^k [with all the pi terms]
    # omega is C^{-1}D, C = sum Phi^k sum Phi^k.T, D = sum Phi^k (reward + gamma Q(state, a))
    def get_A(self, t, reg_param=1e-3):
        X_mat = self.features[t]
        num_basis_funcs = X_mat.shape[1]
        hat_dS2 = (self.delta_S_hat[:, t] ** 2).reshape(-1, 1)
        A_mat = np.dot(X_mat.T, X_mat * hat_dS2) + reg_param * np.eye(num_basis_funcs)
        
        return A_mat

    def get_B(self, t, Pi_hat):
        # This is part (a); in part (b) and (c) we pass the whole Pi_hat in and only operate on each index. 
        coef = 0.00 # or maybe 1.0/(2 * gamma * risk_lambda)
        tmp = Pi_hat * self.delta_S_hat[:, t]
        X_mat = self.features[t]
        B_vec = np.dot(X_mat.T, tmp)
        return B_vec

    def get_C(self, t):
        X_mat = self.features[t]
        num_basis_funcs = X_mat.shape[1]
        C_mat = np.dot(X_mat.T, X_mat) + self.reg_param * np.eye(num_basis_funcs)
        return C_mat

    def get_D(self, t, Q):
        X_mat = self.features[t]
        D_vec = np.dot(X_mat.T, self.rewards[:, t] + self.gamma * Q[:, t + 1])
        return D_vec

    # get the intrinsic vakue
    # CP: call or put
    def seed_intrinsic(self, strike = None, cp = 'P'): 
        if strike is not None:
            self.strike = strike

        if cp == 'P':
            self.option = np.maximum(self.strike - self.S[:, -1], 0)
            self.intrinsic = np.maximum(self.strike - self.S, 0)
        elif cp == 'C':
            self.option = np.maximum(self.S[:, -1] - self.strike, 0)
            self.intrinsic = np.maximum(self.S - self.strike, 0)

        self.positions[:, -1] = self.intrinsic[:, -1]

    # This is part (a)
    def roll_backward(self):
        for t in range(self.num_steps - 1, -1, -1):
            # determine the expected portfolio val at the next time node. 
            pi_next = self.positions[:, t + 1] + self.opt_hedge[:, t + 1] * self.S[:, t + 1]
            pi_hat = pi_next - np.mean(pi_next)

            A_mat = self.get_A(t)
            B_vec = self.get_B(t, pi_hat)
            phi = np.dot(np.linalg.inv(A_mat), B_vec)
            self.opt_hedge[:, t] = np.dot(self.features[t], phi)

            mult_term = (self.positions[:, t + 1] + (self.opt_hedge[:, t+1] - self.opt_hedge[:, t]) * self.S[:, t + 1])
            self.positions[:, t] = np.exp(-self.r * self.dt) * + mult_term

        # Calculate the initial portfolio value. 
        initPortfolioVal = self.positions[:, 0] + self.opt_hedge[:, 0] * self.S[:, 0]

        # only the second half of the paths. 
        optionVal = np.mean(initPortfolioVal)
        optionValVar = np.std(initPortfolioVal)
        delta = np.mean(self.opt_hedge[:, 0])
        return optionVal, delta, optionValVar

    # Terminal payoff of stock price and K: max(K - st_pr, 0)
    # Here's where the European characteristic comes in: only the last part of the stocks matter. 
    # For American, we can exercise at each step. 
    def compute_pi(self):
        if self.option_type == "put":
            self.pi[:, -1] = np.maximum(self.strike - self.S[:, -1], 0)
        else:
            self.pi[:, -1] = np.maximum(self.S[:, -1] - self.strike, 0)
        # self.pi[:, -1] = self.S[:, -1].apply(lambda x: max(self.strike - x, 0))
        self.pi_hat[:, -1] = self.pi[:, -1] - np.mean(self.pi[:, -1])
        self.opt_hedge[:, -1] = 0
        for t in range(self.num_steps - 1, -1, -1):
            A_mat = self.get_A(t) # reg_param?
            B_vec = self.get_B(t, self.pi_hat[:, t + 1])
            phi = np.dot(np.linalg.inv(A_mat), B_vec)
            self.opt_hedge[:, t] = np.dot(self.features[t], phi)
            val_hold = self.gamma * (self.pi[:, t + 1] - self.opt_hedge[:, t] * self.delta_S[:, t])
            if self.stock_type == "European":
                self.pi[:, t] = val_hold
            elif self.stock_type == "American":
                val_exercise =  np.maximum(self.strike - self.S[:, t], 0) if self.option_type == "put" else np.maximum(self.S[:, t] - self.strike, 0)
                self.pi[:, t] = np.maximum(val_hold, val_exercise)
            self.pi_hat[:, t] = self.pi[:, t] - np.mean(self.pi[:, t])

    def get_rewards(self):
        self.rewards[:, -1] = -self.risk_lambda * np.var(self.pi[:, -1])
        for t in range(self.num_steps):
            self.rewards[:, t] = self.gamma * self.opt_hedge[:, t] * self.delta_S[:, t] - self.risk_lambda * np.var(self.pi[:, t])

    # Q-Learning helper, but only for European options. 
    def qlearn_helper(self):
        self.Q[:, -1] = -self.pi[:, -1] - self.risk_lambda * np.var(self.pi[:, -1])
        for t in range(self.num_steps - 1, -1, -1):
            C_mat = self.get_C(t)
            D_vec = self.get_D(t, self.Q)
            omega = np.dot(np.linalg.inv(C_mat), D_vec)
            self.Q[:, t] = np.dot(self.features[t], omega)

    # Now for the real Q Learning. 
    def qlearn(self):
        self.compute_pi() # Compute the portfolio val, which also include computing the A and B matrices. 
        self.get_rewards() # the R matrix. 
        self.qlearn_helper()
        return -self.Q

if __name__ == "__main__":
    # Okay try to make it consistent, see if we can replicate. 
    np.random.seed(42)
    
    # def __init__(self, S0, strike, mu, sigma, r, T, num_steps, num_paths):
    strike = 100
    S0 = 100      # initial stock price
    mu = 0.05     # drift
    sigma = 0.15  # volatility
    r = 0.03      # risk-free rate
    risk_lambda = 0.001
    T = 1         # maturity
    num_paths = 100
    num_periods = 24
    hMC = DiscreteBS(S0, strike, mu, sigma, r, T, num_periods, num_paths, risk_lambda)
    hMC.gen_basis(12)
    C_QLBS = hMC.qlearn()
    step_size = num_paths // 10
    idx_plot = np.arange(step_size, num_paths, step_size)
    for idx in idx_plot:
        plt.plot(C_QLBS[idx, :])
    plt.xlabel('Time steps')
    plt.title('QLBS Option Price')
    plt.savefig('qlbs.png')
    plt.clf()
    #hMC.seed_intrinsic()
    #option_val, delta, option_val_variance = hMC.roll_backward()
    #bs_call_value = bs_put(0, S0, K=strike, r=r, sigma=sigma, T=T)
    #print('Option value = ', option_val)
    #print('Option value variance = ', option_val_variance)
    #print('Option delta = ', delta)
    #print('BS value', bs_call_value)

