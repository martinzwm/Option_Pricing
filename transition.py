import numpy as np

class BrownianMotion:
    ''' Brownian Motion (Wiener Process) with optional drift.
        Input:
            S0 [float]: current value of stock
            T [float]: expiration date
            M [int]: number of mesh grid in time domain
            r [float]: riskfree rate (i.e. similar to discount factor)
            sigma [float]: volatility factor in diffusion term 
            N [int]: number of simulation in MC
    '''
    def __init__(self, S0, r, sigma, T, M, N):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.M = M
        self.N = N
        self.time_unit = self.T / float(self.M)
    
    def simulate(self, seed=123):
        """ Returns MC price matrix rows: time columns: price-path simulation """
        np.random.seed(seed)
        MCprice_matrix = np.zeros((self.M + 1, self.N), dtype=np.float64)
        MCprice_matrix[0,:] = self.S0
        for t in range(1, self.M + 1):
            brownian = np.random.standard_normal( int(self.N / 2) )
            brownian = np.concatenate((brownian, -brownian))
            MCprice_matrix[t, :] = (MCprice_matrix[t - 1, :]
                                  * np.exp((self.r - self.sigma ** 2 / 2.) * self.time_unit
                                  + self.sigma * brownian * np.sqrt(self.time_unit)))
        return MCprice_matrix