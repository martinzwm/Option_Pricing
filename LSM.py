import numpy as np
from transition import BrownianMotion
import matplotlib.pyplot as plt

class AmericanOptionsLSMC(object):
    """ Class for American options pricing using Longstaff-Schwartz (2001):
        Modified from https://github.com/jpcolino/IPython_notebooks/blob/master/Least%20Square%20Monte%20Carlo%20Implementation%20in%20a%20Python%20Class.ipynb
        Input:
            option_type [string]: type of option (i.e. American, European, Asian)
            S0 [float]: current value of stock
            strike [float]: strike price
            T [float]: expiration date
            M [int]: number of mesh grid in time domain
            r [float]: riskfree rate (i.e. similar to discount factor)
            sigma [float]: volatility factor in diffusion term 
            div [float]: dividend yield
            N [int]: number of simulation in MC
    
    Unitest(doctest): 
    >>> AmericanPUT = AmericanOptionsLSMC('put', 36., 40., 1., 50, 0.06, 0.06, 0.2, 10000, BrownianMotion)
    >>> AmericanPUT.price
    4.4731177017712209
    """

    def __init__(self, option_type, S0, strike, T, M, r, div, sigma, N, transition):
        try:
            self.option_type = option_type
            assert isinstance(option_type, str)
            self.S0 = float(S0)
            self.strike = float(strike)
            assert T > 0
            self.T = float(T)
            assert M > 0
            self.M = int(M)
            assert r >= 0
            self.r = float(r)
            assert div >= 0
            self.div = float(div)
            assert sigma > 0
            self.sigma = float(sigma)
            assert N > 0
            self.N = int(N)
            self.transition = transition(S0, r, sigma, T, M, N)
        except ValueError:
            print('Error passing Options parameters')


        if option_type != 'call' and option_type != 'put':
            raise ValueError("Error: option type not valid. Enter 'call' or 'put'")
        if S0 < 0 or strike < 0 or T <= 0 or r < 0 or div < 0 or sigma < 0:
            raise ValueError('Error: Negative inputs not allowed')

        self.time_unit = self.T / float(self.M)
        self.discount = np.exp(-self.r * self.time_unit)
        self.discount = 0.99


    @property
    def MCprice_matrix(self, seed=123):
        """ Returns MC price matrix rows: time columns: price-path simulation """
        MCprice_matrix = self.transition.simulate(seed)
        return MCprice_matrix

    @property
    def MCpayoff(self):
        """Returns the intrinsic value of American Option at each time step and trajectory"""
        if self.option_type == 'call':
            payoff = np.maximum(self.MCprice_matrix - self.strike,
                           np.zeros((self.M + 1, self.N),dtype=np.float64))
        else:
            payoff = np.maximum(self.strike - self.MCprice_matrix,
                            np.zeros((self.M + 1, self.N),
                            dtype=np.float64))
        return payoff

    @property
    def value_matrix(self):
        """ Returns the expected value at each time step and trajectory
            Output:
                value_matrix [matrix of size M x N, float]: expected value
                decision_matrix [matrix of size M x N, bool]: decision
        """
        value_matrix = np.zeros_like(self.MCpayoff)
        decision_matrix = np.zeros_like(self.MCpayoff)

        # Terminal T
        value_matrix[-1, :] = self.MCpayoff[-1, :]
        decision_matrix[-1, :] = self.MCpayoff[-1, :] > 0

        # Recusion from T-1 to 0 
        for t in range(self.M - 1, -1, -1):
            regression = np.polyfit(self.MCprice_matrix[t, :], value_matrix[t + 1, :] * self.discount, 5)
            continuation_value = np.polyval(regression, self.MCprice_matrix[t, :])
            value_matrix[t, :] = np.where(self.MCpayoff[t, :] > continuation_value,
                                          self.MCpayoff[t, :],
                                          value_matrix[t + 1, :] * self.discount)
            decision_matrix[t, :] = np.where(self.MCpayoff[t, :] > continuation_value, True, False)

        return value_matrix, decision_matrix


    @property
    def decision(self):
        """ Returns the action (i.e. continue or exercise) at each time step and trajectory
            Information is inferred by comparing self.MCpayoff and self.value_vector
        """
        value_exercise = self.MCpayoff
        value_actual = self.value_vector
        decision = value_exercise == value_actual
        return decision


    @property
    def price(self):
        value, _ = self.value_matrix
        return np.sum(value[0,:]) / float(self.N)