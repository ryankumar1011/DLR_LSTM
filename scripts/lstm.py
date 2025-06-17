from datetime import datetime
import numpy as np
import torch
from torch import nn
from evaluation import get_predictions

# make stateless and stateful models both

class SocLSTM(nn.Module):
    def __init__(self, input_size=None, output_size=None, num_layers=None, hidden_size=None, dropout=None):
        super(SocLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Fully connected layers applied to each time step
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)

        self.activation = nn.SELU()

    def forward(self, x):
        # First LSTM (return sequences)
        x, _ = self.lstm(x)

        # Apply FC layers
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # Linear activation for final layer

        return x

def build_model(train_loader, input_size, output_size, num_layers, hidden_size, dropout,
                num_epochs, learning_rate, test_loader=None, path=None):
    # print start time
    start_time = datetime.now()
    print(f'Training Start Time : {start_time}')

    # initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SocLSTM(input_size=input_size, output_size=output_size,
                    hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    model = model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # calculate loss
        avg_train_loss = epoch_loss / len(train_loader)

        # printing train and test loss
        if(epoch + 1) % 5 == 0:
            # print epoch number
            print(f'Epoch [{epoch + 1}/{num_epochs}]:')

            # print average test loss for this epoch
            print(f'Train Loss : {avg_train_loss:.6f}')

            if test_loader:
                # Evaluate on test set
                model.eval()
                with torch.no_grad():
                    test_loss = 0
                    for batch_x, batch_y in test_loader:
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        test_loss += loss.item()

                avg_test_loss = test_loss / len(test_loader)
                print(f'Test Loss: {avg_test_loss:.6f}')

                # get np array of predictions for each batch
                predictions, targets = get_predictions(model, test_loader)

                # concat all predictions
                predictions = np.concatenate(predictions)
                targets = np.concatenate(targets)

                # print average test loss for this epoch
                mse = np.mean((predictions - targets) ** 2)
                print(f'Test Loss (MSE): {mse}')

    # if path provided, save model to path
    if path:
        torch.save(model.state_dict(), path)
        print(f'Model state saved to lstm_model.pth {path}')

    # print end time
    end_time = datetime.now()
    print(f'Training End Time : {end_time}')

    # print time it took to train model
    training_time = end_time - start_time
    print(f'Training time : {training_time}')

    return model
'''

class SocLSTM(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, num_layers=None, dropout=None):
        super(SocLSTM, self).__init__()

        # First LSTM layer (returning sequences)
        self.lstm1 = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             batch_first=True)

        # Second LSTM layer (returning sequences as well now)
        self.lstm2 = nn.LSTM(input_size=hidden_size,
                             hidden_size=hidden_size,
                             batch_first=True)

        # Fully connected layers applied to each time step
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.activation = nn.SELU()


        super(SocLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Fully connected layers applied at every time step
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.activation = nn.SELU()

'''
