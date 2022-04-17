class LSM(object):
    def __init__(self, option_type, S0, strike, T, M, r, sigma, div):
        """
            Input:
                option_type [string]: type of option (i.e. American, European, Asian)
                S0 [float]: current value of stock
                strike [float]: strike price
                T [float]: expiration date
                M [int]: number of mesh grid in time domain
                r [float]: riskfree rate (i.e. similar to discount factor)
                div [float]: dividend yield
        """
        self.parameters = {} # use appropriate date structure to keep track of parameters
        assert(option_type in ["American", "European", "Asian"]), "option type not supported yet"
        self.option_type = option_type
        self.S_now = S0 # Price
        self.strike = strike
        self.T = T
        self.M = M
        self.r = r
        self.sigma = sigma
        self.div = div
        self.N = T # we can always change that later 
        self.dt = T / N
        if option_type == "European":
            self.S_min = 0
            self.S_max = 8 * sigma * np.sqrt(T)
            self.dS = (self.S_max - self.S_min) / M
            self.S_arr = np.linspace(self.S_min, self.S_max, M + 1) # array of stock prices
            self.t = np.linspace(0, T, self.N + 1) # array of time scales
        self.v = np.empty((self.N + 1, self.M + 1))
        self.init_v()

    def init_v(self):
        """
            Here, set boundary conditions:
            V(t, S_min)=0 (bottom boundary condition)
            V(t, S_max) = S_max - exp(-r(1-t))K (top boundary condition)
            V(1, S) = max(S - K, 0) (final boundary condition)
        """
        self.v[:, 0] = 0 # Bottom boundary condition. 
        self.v[:, -1] = self.S_max - np.exp(-self.r * (1 - self.t)) * K # Top boundary condition. 
        self.v[-1, :] = np.max(0, self.S_arr - self.strike) # Final boundary condition.

    def get_first_order(self, t):
        # Ignoring the two boundaries, for now. 
        # More like, we only need to worry about the terms in the middle. 
        return (self.v[t, 2:] - self.v[t, :-2]) / (2 * self.dS)

    def get_second_order(self, t):
        return (self.v[t, 2:] - 2 * self.v[t, 1:-1] + self.v[t, :-2]) / (self.dS ** 2)

    def backward_step(self, t):
        """
            According to the article, we have V_t + 1/2 * (sigma**2) * (S**2) * V_SS + r * S * V_S - r * V = 0, 
            where subscript means the partial differential term. 
            Here, S is stock price, V is the value. 

            Thus this motivates the following: 
            v_{t-1} = v_t - 1/2 * (sigma**2) * (S**2) * V_SS - r * S * V_S + r * V
        """
        first_ord = self.get_first_order(self, t)
        second_ord = self.get_second_order(self, t)
        S_ = self.S_arr[1 : -1]
        # Term1 = first order term, term2 = second order term
        term2 = (self.sigma ** 2) * (S_ ** 2) * second_ord / 2
        term1 = r * S_ * first_ord
        rV = r * self.v[1 : -1]
        # The algorithm includes the "W" term, but it looks like we already included that in our computation of the first and second order terms. 
        W = np.zeros(self.M - 1)
        W[0] = -self.sigma**2 * S[0]**2/(2* self.dS**2 ) + r*self.S[0]/(2*dS)
        self.v[t - 1, 1 : -1] = self.v[1, i:-1] - self.dt * (term2 + term1 - rV) 

    def simulate(self):
        """ Given current price, strike, T and other parameters of the option, find its valuation by solving PDE
            Some reference: https://towardsdatascience.com/option-pricing-using-the-black-scholes-model-without-the-formula-e5c002771e2f
            or https://github.com/wessle/option_pricer/blob/master/fin_diffs.py

        """
        for i in range(self.N, 0, -1):
            self.backward_step(i)
        return self.v, self.t, self.S
