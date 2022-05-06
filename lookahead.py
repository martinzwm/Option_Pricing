import numpy as np
from LSM import AmericanOptionsLSMC
from transition import BrownianMotion

def get_ktuples(lst, N, k):
    if k == 0:
        return [[]]
    answer = []
    for (i, item) in enumerate(lst):
        lst_small = lst[:i] + lst[i + 1:]
        subans = [item] + get_ktuples(lst_small, N, k - 1)
        answer.append(subans)
    answer.append([N] + get_ktuples(lst, N, k - 1)
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
        prices_filtered = prices[state]
        if np.sum(state) > 0 and min(prices_filtered) < goal:
            act = ex_action_set[state][np.argmin(prices_filtered)]
            state[act] = 0
            val += min(0, strike - prices[action])
        else:
            act = N
    return val

class Lookahead():
    def __init__(self, amc, goal = None, seed = 123):
        self.amc = amc
        self.T = amc.T
        self.M = amc.M
        self.N = amc.N # This is important: the number of stocks we have. 
        self.prices = amc.MCprice_matrix(seed)
        self.goal = goal if goal is not None else self.strike
        self.strike = strike
        self.state = np.ones((self.N))
        self.action_set = np.arange((self.N + 1))
        self.ex_action_set = np.arange((self.N))
        self.t = 0
        self.val = 0

    # Here, we make one step of simulation
    def simulate_one(self, cur_price):
        brownian = np.random.standard_normal( int(self.N / 2) )
        brownian = np.concatenate((brownian, -brownian))
        answer = cur_price * np.exp((self.r - self.sigma ** 2 / 2.) * self.dt
               + self.sigma * brownian * np.sqrt(self.dt))

    def simulate(self, start_price, num_steps):
        path = np.empty((self.N, num_steps))
        cur_price = start_price
        for i in range(num_steps):
            path[i] = self.simulate_one(cur_price)
            cur_price = path[i]
        return path

    def baseline(self):
        for t in range(self.t, self.T+1):
            prices = self.prices[self.state]
            if np.sum(self.state) > 0 and min(prices) < self.goal:
                action = self.ex_action_set[self.state][np.argmin(prices)]
                self.state[action] = 0
                self.val += min(0, self.strike - self.prices[action])
                # Assert self.prices[action] = prices[np.argmin(prices)]
            else:
                action = self.N
            
    def lookahead(self, num_steps = 1):
        # Get all the k-tuples, even if some doesn't make sense. 
        # It's likely that we won't even get to do k >= 3 since it will be computationally expensive by then. 
        # Unless we keep the # stocks small. 
        num_steps = min(num_steps, self.T - self.t)
        k_tup = get_ktuples(self.ex_action_set[self.state].tolist(), self.N, num_steps)
        val_map = {}
        for tup in k_tup:
            # Part 1: try the k actions. 
            cur_state = self.state.copy()
            cur_val = self.val
            for (i, act) in enumerate(tup):
                if act < self.N:
                    cur_state[act] = 0
                    cur_val += min(0, self.strike - self.prices[self.t + i - 1, act])
            t_now = self.t + self.num_steps
            t_left = self.T - t_now
            est_val = []
            for g in path_number:
                path = self.simulate(self.prices[self.t + num_steps - 1], t_left)
                val_add = baseline_custom(path, cur_state.copy(), self.strike, self.goal, self.N, ex_action_set)
                new_val = cur_val + val_add
                est_val.append(new_val)
             val_map[tup] = sum(est_val) / len(est_val)
  
        idx = max(val_map, key=val_map.get)
        return idx
  
if __name__ == "__main__":
    S0 = 36.0
    strike = 40.0
    T = 1
    M = 50
    N = 20
    r = 0.06
    div = 0.00
    sigma = 0.2
    amc = AmericanOptionsLSMC('put', S0, strike, T, M, r, div, sigma, N, BrownianMotion)
    la = Lookahead(amc)
    la.baseline()
