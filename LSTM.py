from transition import BrownianMotion
from utility import *

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

    # # Generate data
    # data = gen_data()

    # Read real data
    data = read_data()

    # Create dataloaders
    lookback = 5
    train_set, test_set = data_load(data, lookback)
    train_set = SimDataset(train_set)
    test_set = SimDataset(test_set)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False
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