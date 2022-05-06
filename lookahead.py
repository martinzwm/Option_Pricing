import numpy as np
import copy
import matplotlib.pyplot as plt
from transition import BrownianMotion
from tqdm import tqdm

def get_ktuples(lst, N, k):
    if k == 0:
        return [[]]
    answer = []
    for (i, item) in enumerate(lst):
        lst_small = lst[:i] + lst[i + 1:]
        sublst = get_ktuples(lst_small, N, k - 1)
        for subsublst in sublst:
            answer.append([item] + subsublst)

    sublst = get_ktuples(lst, N, k - 1)
    for lst in sublst:
        answer.append([N] + lst)
    return answer
    
def baseline_custom(prices, state, strike, goal, N, ex_action_set):
    """
        Args:
            prices: T x N, T = time frame, N = num stocks
        Returns:
            Value function based on this trajectory. 
    """
    T, N = prices.shape
    val = 0
    for t in range(T):
        prices_filtered = prices[t, state]
        if np.sum(state) > 0 and min(prices_filtered) < goal:
            act = ex_action_set[state][np.argmin(prices_filtered)]
            state[act] = 0
            val += max(0, strike - prices[t, act])
        else:
            act = N
    return val

class Lookahead():
    def __init__(self, S0, strike, r, sigma, T, M, 
                       num_stocks, num_repeats, goal = None, seed = 123):
        self.T = T
        self.M = M
        self.r = r
        self.dt = self.T / self.M
        self.gamma = np.exp(-self.r * self.dt)
        self.strike = strike
        self.sigma = sigma
        self.goal = goal if not (goal is None) else self.strike
        self.num_stocks = num_stocks
        self.num_repeats = num_repeats
        self.seed = seed
        self.brownian = BrownianMotion(S0, r, sigma, T, M, N = num_stocks * num_repeats)
        # self.state = np.ones((self.num_stocks)).astype(np.bool_)
        self.action_set = np.arange((self.num_stocks + 1))
        self.ex_action_set = np.arange((self.num_stocks))
        # self.t = 0
        # self.val = 0

    # Here, we make one step of simulation
    def simulate_one(self, cur_price):
        brownian = np.random.standard_normal( int(self.num_stocks / 2) )
        brownian = np.concatenate((brownian, -brownian))
        answer = cur_price * np.exp((self.r - self.sigma ** 2 / 2.) * self.dt
               + self.sigma * brownian * np.sqrt(self.dt))
        return answer

    def simulate(self, start_price, num_steps):
        path = np.empty((num_steps, self.num_stocks))
        cur_price = start_price
        for i in range(num_steps):
            path[i] = self.simulate_one(cur_price)
            cur_price = path[i]
        return path

    def baseline(self):
        
        cur_val = self.val
        all_vals = np.empty(num_repeats)
        prices = np.hsplit(self.brownian.simulate(self.seed), num_repeats)
        for (i, prices_i) in enumerate(prices):
            self.state = np.ones((self.num_stocks)).astype(np.bool_)
            self.val = cur_val
            for t in range(self.t, self.M+1):
                price_ftr = prices_i[t, self.state]
                best_price = min(price_ftr)
                if np.sum(self.state) > 0 and best_price < self.goal:
                    action = self.ex_action_set[self.state][np.argmin(price_ftr)]
                    self.state[action] = 0
                    self.val += max(0, self.strike - best_price) * (self.gamma ** t)
                else:
                    action = self.num_stocks
            # Finally, anything that's not actioned will be exercised as time ends. 
            prices = prices_i[-1, self.state]
            extra = np.maximum(self.strike - prices, np.zeros_like(prices)) * (self.gamma ** self.M)
            self.val += np.sum(extra)
            all_vals[i] = self.val / self.num_stocks
        return np.mean(all_vals)
            
    def lookahead_helper(self, t, prices, state, num_steps):
        """
            Args:
                prices: T x (M + 1)
                num_steps: number of lookahead steps. 
            Returns:
                series of actions. 
        """
        # Get all the k-tuples, even if some doesn't make sense. 
        # It's likely that we won't even get to do k >= 3 since it will be computationally expensive by then. 
        # Unless we keep the # stocks small. 
        num_steps = min(num_steps, self.M + 1 - t)
        k_tup = get_ktuples(self.ex_action_set[state].tolist(), self.num_stocks, num_steps)
        val_map = {}
        path_number = 10
        t_now = t + num_steps
        t_left = self.M + 1 - (t_now)
        all_paths = [self.simulate(prices[t + num_steps - 1], t_left) for _ in range(path_number)]
        for tup in k_tup:
            # Part 1: try the k actions. 
            cur_state = state.copy()
            cur_val = 0
            for (i, act) in enumerate(tup):
                if act < self.num_stocks:
                    cur_state[act] = 0
                    cur_val += max(0, self.strike - prices[t + i - 1, act])
            est_val = []
            for path in all_paths:
                path = self.simulate(prices[t + num_steps - 1], t_left)
                val_add = baseline_custom(path, cur_state.copy(), self.strike, 
                                          self.goal, self.num_stocks, self.ex_action_set)
                new_val = cur_val + val_add
                est_val.append(new_val)
            val_map[tuple(tup)] = sum(est_val) / len(est_val)
        idx = max(val_map, key=val_map.get)
        return idx

    def lookahead(self, num_steps):
        all_vals = np.empty(num_repeats)
        prices = np.hsplit(self.brownian.simulate(self.seed), num_repeats)
        for (ind, prices_i) in tqdm(enumerate(prices)):
            val = 0
            t = 0
            state = np.ones((self.num_stocks)).astype(np.bool_) # Assume start from 0. 
            while t < self.M + 1:
                actions = self.lookahead_helper(t, prices_i, state, num_steps)
                for act in actions:
                    if act < self.num_stocks: 
                        state[act] = 0
                        pr = prices_i[t, act]
                        val += max(0, self.strike - pr) * (self.gamma ** t) 
                    t += 1
            pr_left = prices_i[-1, state]
            extra = np.maximum(self.strike - pr_left, np.zeros_like(pr_left)) * (self.gamma ** self.M)
            val += np.sum(extra)
            all_vals[ind] = val / self.num_stocks
        return np.mean(all_vals)

  
def simulate_baseline(S0, strike, T, M, r, sigma, num_stocks, num_repeats):
    goals = np.linspace(30.00, 40.00, 101)
    val = np.zeros_like(goals)
    for (i, goal) in tqdm(enumerate(goals)):
        la = Lookahead(S0, strike, r, sigma, T, M, num_stocks, num_repeats, goal)
        val[i] = la.baseline()
    plt.plot(goals, val)
    plt.savefig("la.png")
    plt.clf()

if __name__ == "__main__":
    # Okay let's see how lookahead will fare. 
    S0 = 36.0
    strike = 40.0
    T = 1
    M = 50
    num_stocks = 20
    num_repeats = 10
    r = 0.06
    div = 0.00
    sigma = 0.2
    goal = 32.0
    la = Lookahead(S0, strike, r, sigma, T, M, num_stocks, num_repeats, goal)
    print(la.lookahead(2))
    
