from transition import BrownianMotion
from LSM import AmericanOptionsLSMC
from Optimal import Optimal
from utility import *
from LSTM import *
import matplotlib.pyplot as plt
import matplotlib

def test_LSM():
    """ American Put LSM method
    """
    AmericanPUT = AmericanOptionsLSMC(option_type='put', S0=36, strike=40, T=1, M=50,
                                    r=0.06, div=0, sigma=0.2, N=1000, transition=BrownianMotion)
    print('Price for LSM: ', AmericanPUT.price)


def test_Optimal():
    """ Optimal action
    """
    Opt = Optimal(option_type='put', S0=36, strike=40, T=1, M=50,
                                    r=0.06, div=0, sigma=0.2, N=1000, transition=BrownianMotion)
    print('Price for Optimal: ', Opt.price)

def test_traj_gen():
    """ Generate trajectories
    """ 
    S0, r, sigma, T, M, N = 36, 0.02, 0.2, 1, 50, 10000
    traj_gen = BrownianMotion(S0, r, sigma, T, M, N)
    traj1 = traj_gen.simulate()[:,10] # take a look at the 10th trajectory
    print(traj1)
    # plt.plot(traj1)
    # plt.savefig("test.png")


def test_data_gen():
    X, S0, r, sigma, T, M, N, transition = 40, 36, 0.06, 0.2, 1, 100, 10, BrownianMotion
    data = gen_traj(X, S0, r, sigma, T, M, N, transition)
    # print(data.shape)
    return data


def test_dataloader():
    """ Create dataloader
    """
    # Generate data
    X, S0, r, sigma, T, M, N, transition = 40, 36, 0.06, 0.2, 1, 100, 10, BrownianMotion
    data = gen_traj(X, S0, r, sigma, T, M, N, transition)
    # Create dataloader
    train_set, test_set = data_load(data, lookback=20)
    print(train_set.size(), test_set.size())


def compare():
    """ This function compares the simulation trajectory (i.e. stock price under Brownian motion)
        and real market price for S&P 500 index.
    """
    # Real data
    stock_data = pd.read_excel('Data/stock_price.xlsx', skiprows=17,  nrows=252, usecols='A:B')
    time = stock_data["Exchange Date"].values[::-1]
    stock_real = stock_data["Close"].values[::-1]

    # Simulated data
    S0, r, sigma, T, M, N = stock_real[0], 0.06, 0.2, 1, 365, 10
    time_pred = matplotlib.dates.date2num(time[0]) + np.linspace(0, 364, 365)
    traj_gen = BrownianMotion(S0, r, sigma, T, M, N)
    trajs = traj_gen.simulate()
    traj_mean = np.mean( trajs[:-1, :], axis=1 )
    traj_std = np.std( trajs[:-1, :], axis=1 )

    # Plot to compare
    plt.figure()
    plt.plot(time_pred, traj_mean, time, stock_real)
    plt.legend(["Simulated", "Actual"])
    plt.fill_between(time_pred, traj_mean - traj_std, traj_mean + traj_std,
                     alpha=0.3)
    plt.xlabel("Time"); plt.ylabel("S&P 500")
    plt.savefig("Plots/sim_vs_real_stock.png")

    # Plot simulated trajectories
    plt.figure()
    plt.plot(time, stock_real)
    for i in range(N):
        plt.plot(time_pred, trajs[:-1, i])
    plt.xlabel("Time"); plt.ylabel("S&P 500")
    plt.savefig("Plots/simulated_trajectories.png")




if __name__ == "__main__":
    test_LSM()
    # test_Optimal()