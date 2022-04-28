from transition import BrownianMotion
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *

# Model
class LSTM(nn.Module):
    def __init__(self, in_channels=5, num_classes=1, fc_layer_sizes=[64,128], lstm_hidden_size=20):
        super(LSTM, self).__init__()
        # fc layers
        self.fc_layer1 = self._fc_layer_set(in_channels, fc_layer_sizes[0])
        self.fc_layer2 = self._fc_layer_set(fc_layer_sizes[0], fc_layer_sizes[1])
        flattened_size = fc_layer_sizes[1]
        # LSTM layers
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_cell = nn.LSTMCell(flattened_size, lstm_hidden_size)
        self.i2o = nn.Linear(flattened_size + lstm_hidden_size, lstm_hidden_size)
        self.drop=nn.Dropout(p=0.5)
        # final fc layer
        self.fc_final = nn.Linear(lstm_hidden_size, num_classes)

    # Block of fc layers
    # Architecture: (fc, batchnorm, relu, dropout)
    def _fc_layer_set(self, in_c, out_c):
        # first block
        fc_layer = nn.Sequential(
        nn.Linear(in_c, out_c),
        # nn.BatchNorm1d(num_features=out_c),
        nn.LeakyReLU(),
        nn.Dropout(p=0.5)
        )
        return fc_layer
    
    # Use fc to extract features
    def _fc(self,x):
        out = self.fc_layer1(x)
        out = self.fc_layer2(out)
        # out = out.view(out.size(0), -1) # Flatten it out
        return out
    
    # Use LSTM to look at extracted features from CNN
    def _lstm(self, input, hidden_and_cell):
        hidden = hidden_and_cell[0]
        combined = torch.cat((input, hidden), 1)
        hidden_and_cell = self.lstm_cell(input, hidden_and_cell)
        combined = self.drop(combined)
        output = self.i2o(combined)
        return output, hidden_and_cell

    def initHidden(self, input):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = input.size(0)
        return (torch.zeros(batch_size, self.lstm_hidden_size).to(device), torch.zeros(batch_size, self.lstm_hidden_size).to(device))

    def forward(self, x):
        lstm_hidden = self.initHidden(x)
        for i in range(x.size(1)): # lstm loop
            ft = self._fc(x[:,i]) # extract state features with fc
            out, lstm_hidden = self._lstm(ft, lstm_hidden)
        # fully-connected layers
        out = self.fc_final(out)
        return out


# Data generator
def data_gen(X, S0, r, sigma, T, M, N, transition):
    """ Generate trajectory data
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
    """
    train_set, test_set = [], []
    
    for traj in data:
        train_set_traj, test_set_traj = traj_load(traj, lookback, train_test_split)
        train_set.append(train_set_traj)
        test_set.append(test_set_traj)
    
    # Convert to numpy arrays
    train_set, test_set = np.array(train_set), np.array(test_set)
    # Convert to torch tensors
    train_set = torch.from_numpy(train_set).type(torch.Tensor)
    test_set = torch.from_numpy(test_set).type(torch.Tensor)
    # Merge the 1st and 2nd dimensions, i.e. model doesn't need to know which trajectory we are on
    train_set = train_set.view(-1, lookback, train_set.size()[-1])
    test_set = test_set.view(-1, lookback, test_set.size()[-1])
    return train_set, test_set


def traj_load(traj, lookback, train_test_split=0.8):
    """ Prepare LSTM data from raw (single) trajectory data.
        Helper function to load_data()
        Input:
            traj [matrix of size M x 6]: M is the number of timesteps; at each time step
                                         there are 5 state variables and 1 output price
            lookback [scalar]: number of lookback steps
            train_test_split [scaler]: ratio of train : test dataset
        Output:
            train_set [matrix of size len(traj)*train_test_split x lookback x 5]: training data
            test_set: analogous to train_set
    """
    M = traj.shape[0]

    # Prepare path into LSTM data format
    data = []
    for i in range(M - lookback):
        data.append(traj[i : i + lookback])
    data = np.array(data)
    np.random.shuffle(data)

    # Split train and test set
    train_size = int(np.round(train_test_split * data.shape[0]))
    train_set = data[:train_size, :, :]
    test_set = data[train_size:, :, :]
    return [train_set, test_set]


# Training
def train():
    """ Train the model
    """
    # Use cpu or gpu acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # larger batch size can be used for gpu (faster training)
    if device == torch.device('cuda'):
        batch_size = 64
    else:
        batch_size = 8

    # Create model
    model = LSTM()
    # print(lstm)
    # print(len(list(lstm.parameters())))
    # for i in range(len(list(lstm.parameters()))):
    #     print(list(lstm.parameters())[i].size())

    # Generate data
    X, S0, r, sigma, T, M, N, transition = 40, 36, 0.06, 0.2, 1, 100, 800, BrownianMotion
    data = data_gen(X, S0, r, sigma, T, M, N, transition)

    # Create dataloaders
    lookback = 20
    train_set, test_set = data_load(data, lookback)
    train_set = SimDataset(train_set)
    test_set = SimDataset(test_set)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True
    )

    # Train parameters
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    hist = np.zeros(num_epochs)

    # Train model
    for t in range(num_epochs):       
        for i, sample in enumerate(train_loader):
            model.train() # set mode to training, for dropout layers
            x_train = sample[:, :-1, :]
            y_train = sample[:, -1, 1] # stock price in the end
            
            # Forward pass
            y_train_pred = model(x_train).view(-1) 
            loss = loss_fn(y_train_pred, y_train)

            # Zero out gradient, else they will accumulate between epochs
            optimiser.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimiser.step()
        
        if t % 1 == 0 and t !=0:
            print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()


# Testing
def test():
    pass


if __name__ == "__main__":
    train()