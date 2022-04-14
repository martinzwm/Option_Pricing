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
        raise NotImplementedError

    def simulate():
        """ Given current price, strike, T and other parameters of the option, find its valuation by solving PDE
            Some reference: https://towardsdatascience.com/option-pricing-using-the-black-scholes-model-without-the-formula-e5c002771e2f
            or https://github.com/wessle/option_pricer/blob/master/fin_diffs.py
        """
        raise NotImplementedError