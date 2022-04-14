class LSM(object):
    def __init__(self, option_type, S0, strike, T, M, r, sigma, div, N):
        """
            Input:
                option_type [string]: type of option (i.e. American, European, Asian)
                S0 [float]: current value of stock
                strike [float]: strike price
                T [float]: expiration date
                M [int]: number of mesh grid in time domain
                r [float]: riskfree rate (i.e. similar to discount factor)
                div [float]: dividend yield
                
                N [int]: number of simulations

            Some reference: https://github.com/jpcolino/IPython_notebooks/blob/master/Least%20Square%20Monte%20Carlo%20Implementation%20in%20a%20Python%20Class.ipynb
            or https://cran.r-project.org/web/packages/LSMRealOptions/vignettes/LSMRealOptions.html#:~:text=The%20Least%2DSquares%20Monte%20Carlo,option%20at%20discrete%20observation%20points.
        """
        self.parameters = {} # use appropriate date structure to keep track of parameters
        raise NotImplementedError

    def train():
        """ Given a trajectory object, find fitted parameters based on the trajectory.
        """
        raise NotImplementedError

    def price():
        """ Given current price, strike, and other parameters of option, find its valuation using the fitted parameters
        """
        raise NotImplemented