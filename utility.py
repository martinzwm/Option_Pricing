import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# Data reader
def read_data():
    """ Read real option dataset from excel
    """
    file_name = "Data/sxp.csv"
    df = pd.read_csv(file_name)
    df = pd.DataFrame(df)
    data = []
    
    # Group by symbols (i.e. different options)
    trajs = df.groupby('symbol')
    for traj in trajs:
        option_id, traj_data = traj[0], traj[1]
        # Remove trajectories with less than 10 data points
        if len(traj_data) >= 10:
            # Fields that we need
            days_to_expire = traj_data["days_to_expire"].values
            stock_price = traj_data["stock_price"].values
            riskfree_rate = traj_data["riskfree_rate"].values
            strike_price = traj_data["strike_price"].values / 1000
            bid = traj_data["best_bid"].values
            
            traj_data = np.vstack((days_to_expire, stock_price, riskfree_rate, strike_price, bid))
            traj_data = traj_data.T
            data.append(traj_data)
    
    data = np.array(data, dtype=object)
    return data  


# Data generator
def gen_data():
    """ Generate dataset
        Uses gen_traj() as a helper function. Use it multiple times 
        at different specifications to generate diverse data.
    """
    # X, S0, r, sigma, T, M, N, transition = 40, 36, 0.06, 0.2, 1, 100, 800, BrownianMotion
    pass

def gen_traj(X, S0, r, sigma, T, M, N, transition):
    """ Generate N trajectories based on the specifications (i.e. all other inputs)
        Input:
            X, S0, r, sigma, T [scalar]: specifies the option
            M, N [scalar]: number of time steps, number of trajectories
            transition [class]: a transition function
    """
    brownian = transition(S0, r, sigma, T, M, N)
    data = []
    trajs = brownian.simulate()

    for i in range(trajs.shape[1]):
        traj = trajs[:,i]
        traj = np.vstack( (np.linspace(0, T*365, M+1), 
                        traj, 
                        X*np.ones(M+1), 
                        r*np.ones(M+1), 
                        sigma*np.ones(M+1)) )
        traj = traj.T
        data.append(traj)
    
    data = np.array(data)
    return data


# Dataset
class SimDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# Dataloader
def data_load(data, lookback, train_test_split=0.8):
    """ Prepare LSTM data from a list of trajectories. 
        Uses load_traj() as a helper function
        Input:
            data [matrix of size N x ? x 5]: M is the number of timesteps; at each time step
                                         there are 4 state variables and 1 output price
            lookback [scalar]: number of lookback steps
            train_test_split [scaler]: ratio of train : test dataset
        Output:
            train_set [len(data) * train_test_split x size of traj_load() output]
            test_set: analagous to train_set
    """
    np.random.shuffle(data)
    train_set, test_set = [], []

    train_size = int(np.round(train_test_split * data.shape[0]))
    for traj in data[:train_size]:
        train_set_traj = traj_load(traj, lookback)
        # Model doesn't need to know which trajectory we are on, thus I used extend() instead of append()
        train_set.extend(train_set_traj)
    
    for traj in data[train_size:]:
        test_set_traj = traj_load(traj, lookback)
        test_set.extend(test_set_traj)

    # Convert to numpy arrays
    train_set, test_set = np.array(train_set), np.array(test_set)
    # Convert to torch tensors
    train_set = torch.from_numpy(train_set).type(torch.Tensor)
    test_set = torch.from_numpy(test_set).type(torch.Tensor)
    return train_set, test_set


def traj_load(traj, lookback):
    """ Prepare LSTM data from raw (single) trajectory data.
        Helper function to load_data()
        Input:
            traj [matrix of size ? x 6]: M is the number of timesteps; at each time step
                                         there are 4 state variables and 1 output price
            lookback [scalar]: number of lookback steps
            
        Output:
            data [(len(traj) - lookback) x lookback x 5]: traj data
    """
    M = traj.shape[0]

    # Prepare path into LSTM data format
    data = []
    for i in range(M - lookback):
        data.append(traj[i : i + lookback])
    data = np.array(data)
    return data


if __name__ == "__main__":
    read_data()