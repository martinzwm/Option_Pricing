from configparser import Interpolation
from transition import BrownianMotion
from LSM import AmericanOptionsLSMC
from Optimal import Optimal

import numpy as np
import matplotlib.pyplot as plt


def decision2prob_t(stock, decision, interval=1, min_price=0, max_price=0):
    """ Convert decision (i.e. continue or exercise) at each stock price into 
        probability of exercise at time t
    """
    # Set min and max bound if not specified
    if min_price == 0:
        min_price = int(np.floor( min(stock) ))
    if max_price == 0:
        max_price = int(np.ceil( max(stock) ))
    
    bins = range(min_price, max_price, interval)
    probs = np.zeros((len(bins), 1))

    for i in range(len(bins)-1):
        idx = np.where(np.logical_and(stock >= bins[i], stock < bins[i+1]))
        action = decision[idx]
        if len(action) == 0:
            probs[i] = -1 # invalid
        else:
            probs[i] = np.count_nonzero(action) / len(action)
    return bins, probs
        

def decision2prob(stock, decision, interval=1, min_price=0, max_price=0):
    """ Convert decision (i.e. continue or exercise) at each stock price into 
        probability of exercise at all time. 
        Use decision2prob_t as a helper function.
    """
    i = 20
    bin, prob = decision2prob_t(stock[i, :], decision[i, :], interval=interval, 
                                    min_price=min_price, max_price=max_price)
    # raise NameError("Debug")
    bins, probs = [], []
    for i in range(len(stock)): # traverse time step
        bin, prob = decision2prob_t(stock[i, :], decision[i, :], interval=interval, 
                                    min_price=min_price, max_price=max_price)
        bins.append(bin[:-1]); probs.append(prob[:-1])
            
    bins = np.array(bins); probs = np.array(probs)
    return bins, probs.squeeze()


def decision_plot(model, fig_name):
    """ Can be used to make 2D color plot, x-axis: time, y-axis: stock, 
        color: prob of exercising. Use decision2prob to compute probs.
    """
    
    value, decision = model.value_matrix
    stock = model.MCprice_matrix

    min_price, max_price = int(np.floor( np.min(stock) )), int(np.ceil( np.max(stock) ))
    
    # Convert actions into probabilities
    bins, probs = decision2prob(stock, decision, min_price=min_price, max_price=max_price)
    
    # Plotting
    x = bins[1,:] # stock price
    y = np.linspace(0, 365, 51) # time
    X, Y = np.meshgrid(x, y)
    Z = probs
    plt.figure()
    plt.contourf(X, Y, Z)
    # axs.imshow(interpolation='bilinear')
    plt.xlabel("Stock [$]"); plt.ylabel("Time [day]")
    plt.set_cmap('jet'); plt.colorbar()
    plt.savefig(fig_name)


def visualize():
<<<<<<< Updated upstream
    # LSM
    lsm = AmericanOptionsLSMC(option_type='put', S0=36, strike=40, T=1, M=50,
                                    r=0.06, div=0, sigma=0.2, N=100000, transition=BrownianMotion, stochastic_volatility=True)
    decision_plot(lsm, fig_name="Plots/LSM_decision.png")
=======
    # # LSM
    # lsm = AmericanOptionsLSMC(option_type='put', S0=36, strike=40, T=1, M=50,
    #                                 r=0.06, div=0, sigma=0.2, N=10000, transition=BrownianMotion)
    # decision_plot(lsm, fig_name="Plots/test.png")
>>>>>>> Stashed changes

    # Optimal
    optimal = Optimal(option_type='put', S0=36, strike=40, T=1, M=50,
                                    r=0.06, div=0, sigma=0.2, N=100000, transition=BrownianMotion, stochastic_volatility=True)
    decision_plot(optimal, fig_name="Plots/optimal_decision.png")


if __name__ == "__main__":
    visualize()