import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

data_dir='...\...'

def forecast_RNN(df,model_path=data_dir+"\\rnn_trained_model.pt"):
    '''
    Takes the dataframe as input
    :param df:
    :return: df with the predicted values with RNN prediction
    '''

    # RNN cell
    class RNNCell(torch.nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int):
            """
            Constructor of RNNCell.

            Inputs:
            - input_dim: Dimension of the input x_t
            - hidden_dim: Dimension of the hidden state h_{t-1} and h_t
            """
            super(RNNCell, self).__init__()
            self.linear_x, self.linear_h, self.non_linear = None, None, None
            self.linear_x = torch.nn.Linear(input_dim, hidden_dim)
            self.linear_h = torch.nn.Linear(hidden_dim, hidden_dim)
            self.non_linear = torch.nn.Tanh()
            # self.non_linear=torch.nn.ReLU()

        def forward(self, x_cur: torch.Tensor, h_prev: torch.Tensor):
            """
            Compute h_t given x_t and h_{t-1}.

            Inputs:
            - x_cur: x_t, a tensor with the same of BxC, where B is the batch size and
              C is the channel dimension.
            - h_prev: h_{t-1}, a tensor with the same of BxH, where H is the channel
              dimension.
            """
            h_cur = None
            x_t = self.linear_x(x_cur)
            h_t = self.linear_h(h_prev)
            h_cur = self.non_linear(x_t + h_t)
            return h_cur

    # Single Layer RNN
    class RNN(torch.nn.Module):
        """
        RNN is a single-layer (stack) RNN by connecting multiple RNNCell together in a single
        direction, where the input sequence is processed from left to right.
        """

        def __init__(self, input_dim: int, hidden_dim: int):
            """
            Constructor of the RNN module.

            Inputs:
            - input_dim: Dimension of the input x_t
            - hidden_dim: Dimension of the hidden state h_{t-1} and h_t
            """
            super(RNN, self).__init__()
            self.hidden_dim = hidden_dim
            self.rnncell = RNNCell(input_dim, hidden_dim)

        def forward(self, x: torch.Tensor):
            """
            Compute the hidden representations for every token in the input sequence.

            Input:
            - x: A tensor with the shape of BxLxC, where B is the batch size, L is the squence
              length, and C is the channel dimmension

            Return:
            - h: A tensor with the shape of BxLxH, where H is the hidden dimension of RNNCell
            """
            b = x.shape[0]
            seq_len = x.shape[1]
            # initialize the hidden dimension
            init_h = x.new_zeros((b, self.hidden_dim))
            h_list = []
            for t in range(seq_len):
                x_t = x[:, t, :]
                init_h = self.rnncell.forward(x_t, init_h)
                h_list.append(init_h.unsqueeze(1))
            h = torch.cat(h_list, dim=1)
            return h


    class RNNmodel(nn.Module):
        """
        A 2-layer RNN-based predictor
        """

        def __init__(self, input_dim: int, rnn_hidden_dim: int, output_dim: int):
            """
            Constructor.

            Inputs:
            - input_dim: Input dimension of the sequence
            - rnn_hidden_dim: The hidden dimension of the RNN
            - output_dim: Output dimension
            """
            super(RNNmodel, self).__init__()

            # Define a 3-layer RNN
            self.rnn = nn.RNN(
                input_dim,
                rnn_hidden_dim,
                num_layers=3,  # Add the second layer
                batch_first=True,  # Ensure the input has shape [B, L, C]
                dropout=0.4  # Dropout between RNN layers
            )

            # Fully connected layer for the output
            self.fc = nn.Linear(rnn_hidden_dim, output_dim)

        def init_weights(self):
            """
            Initialize weights for the fully connected layer.
            """
            initrange = 0.1
            self.fc.weight.data.uniform_(-initrange, initrange)
            self.fc.bias.data.zero_()

        def forward(self, x):
            """
            Forward pass of the model.

            Input:
            - x: Tensor with shape [B, L, C].

            Return:
            - y: Tensor with shape [B, O], where O is the output dimension.
            """
            # Pass the input through the RNN
            rnn_out, _ = self.rnn(x)  # rnn_out has shape [B, L, H], where H is the hidden size

            # Use the last time step's output for prediction
            y = self.fc(rnn_out[:, -1, :])  # Extract the output at the last time step
            return y

    #Load RNN model
    model = RNNmodel(input_dim=1, rnn_hidden_dim=512, output_dim=1)
    state_dict = torch.load(data_dir+'train_dataset.pt', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    #model.load_state_dict(torch.load("rnn_trained_model.pt"),map_location=torch.device('cpu'))
    model.eval()  # Set the model to evaluation mode

    # Load the dataset later
    train_dataset = torch.load("train_dataset.pt")
    predicted_temps = np.array([])

    # Extract train_data from train_dataset
    device='cpu'
    train_data = train_dataset[:][0]  # Assuming train_dataset is a tuple (inputs, targets)
    train_data = train_data.float().to(device)  # Convert to tensor and move to device

    # Use the last known sequence
    last_sequence = train_data[-1].unsqueeze(0).unsqueeze(-1).to(device)  # Shape: [1, L, C]

    for _ in range(10 * 12):  # Predict for 10 years (120 months)
        with torch.no_grad():
            next_temp = model(last_sequence)  # Shape: [1, 1]
            predicted_temps = np.append(predicted_temps, next_temp.item())

            # Reshape next_temp to match [1, 1, C] (for concatenation with last_sequence)
            next_temp = next_temp.unsqueeze(-1)  # Shape: [1, 1, C]

            # Update the sequence (remove the oldest timestep, add the new prediction)
            last_sequence = torch.cat((last_sequence[:, 1:, :], next_temp), dim=1)

    # Convert predictions back to original scale
    temperatures = df['Actual Temperature_Air_Global'].iloc[0:2052].values
    max_temp = temperatures.max()
    min_temp = temperatures.min()
    predicted_temps = np.array(predicted_temps) * ((max_temp - min_temp)) + min_temp

    # Step 1: Generate new year and month data for 2021–2031
    years = []
    months = []

    for year in range(2021, 2031):
        for month in range(1, 13):
            years.append(year)
            months.append(month)
    print(len(years), len(months), len(predicted_temps))
    # Step 2: Create the new dataframe for the next 10 years
    new_data = pd.DataFrame({
        "year": years,
        "month": months,
        "Actual Temperature_Air_Global": predicted_temps
    })
    print(predicted_temps)
    # Step 3: Select relevant columns from the original dataframe
    original_data = df.iloc[0:2052][["year", "month", "Actual Temperature_Air_Global"]]

    # Step 4: Append the new data below the original data
    final_df = pd.concat([original_data, new_data], ignore_index=True)

    return final_df


df_loaded = pd.read_csv("df.csv", index_col=0)
#forecast_RNN(df_loaded)
#print(forecast_RNN(df_loaded)[-121:])