from transition import BrownianMotion
from LSM import AmericanOptionsLSMC
from utility import *
from LSTM import *
import matplotlib.pyplot as plt

def test_LSM():
    """ American Put LSM method
    """
    AmericanPUT = AmericanOptionsLSMC(option_type='put', S0=36, strike=40, T=1, M=50,
                                    r=0.06, div=0, sigma=0.2, N=10000, transition=BrownianMotion)
    print('Price: ', AmericanPUT.price)


def test_traj_gen():
    """ Generate trajectories
    """ 
    S0, r, sigma, T, M, N = 36, 0.06, 0.2, 1, 50, 10000
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


if __name__ == "__main__":
    test_LSM()