from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn import MSELoss
from utils import preprocess_pytorch_tensor
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os
import matplotlib.pyplot as plt

import flwr as fl

from main import save_model
from pytorch_model import pytorchCNN, pytorchCNNSpeed, pytorchCNNSpeedThrottle


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(data_dir, test_size, batch_size):
    data = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'))
    print(data.shape)
    #Assign column names
    data.columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']


    #X = data[['center', 'left', 'right', 'speed']].values
    X = data[['center', 'left', 'right']].values
    Y = data[['steering']].values
    #Y = data[['steering', 'throttle']].values

    #Maybe create own shuffling method if scikit learn doesnt allow to return the reordered indexes

    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=test_size, random_state=0)
    print("Loaded data")
    #Preprocess images for dataloaders
    X_train, Y_train = preprocess_pytorch_tensor(X_train, Y_train, data_dir, True)
    X_valid, Y_valid = preprocess_pytorch_tensor(X_valid, Y_valid, data_dir, False)
    #(inputs_train, speeds_train), targets_train = preprocess_pytorch_speed(X_train, Y_train, data_dir, True)
    #(inputs_valid, speeds_valid), targets_valid= preprocess_pytorch_speed(X_valid, Y_valid, data_dir, False)
    print("Finished pre-processing")

    #Convert the training data to torch tensors
    X_train_tensor = X_train #.permute(0, 3, 1, 2)
    X_valid_tensor = X_valid #.permute(0, 3, 1, 2)
    Y_train_tensor = Y_train
    Y_valid_tensor = Y_valid
    print("converted into tensors")

    #make it so that datasets combine the v variables
    #Create TensorDatasets for training and validation data
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, Y_valid_tensor)
    #train_dataset = CustomDataset((inputs_train, speeds_train), targets_train)
    #valid_dataset = CustomDataset((inputs_valid, speeds_valid), targets_valid)
    print("created dataset")

    #Create DataLoaders for training and validation data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    print("created data loaders")

    num_examples = {"trainset" : len(train_dataset), "testset" : len(valid_dataset)}

    return train_loader, valid_loader, num_examples

def train(device, model, train_loader, learning_rate, nb_epoch):

    print("Training model")

    model = model.to(device)

    # Define the loss function and optimizer
    loss_function = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    training_losses = []

    #Training loop
    print("starting training loop")
    for epoch in range(nb_epoch):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:

            #Send input and target to device
            #inputs = tuple(input_tensor.to(device) for input_tensor in inputs)
            #targets = targets.to(device)

            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{nb_epoch}, Loss: {running_loss/len(train_loader)}')
        # Save model
        state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
        }
        save_model(state, "pytorch_client_1")
    average_training_loss = running_loss / len(train_loader)

    #training_losses.append(average_training_loss)

    return average_training_loss

def evaluate(device, model, valid_loader):
    model = model.to(device)

    # Define the loss function and optimizer
    loss_function = MSELoss()

    validation_losses = []
    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            #inputs = tuple(input_tensor.to(device) for input_tensor in inputs)
            #targets = targets.to(device)
            #Send input and target to device
            inputs, targets = inputs.to(device), targets.to(device)
                
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            val_loss += loss.item()

    print(f'Validation Loss: {val_loss/len(valid_loader)}')

    average_validation_loss = val_loss / len(valid_loader)

    #validation_losses.append(average_validation_loss)

    return val_loss, average_validation_loss

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, num_examples):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_examples = num_examples
        self.training_losses = []
        self.validation_losses = []
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        average_training_loss = train(DEVICE, self.model, self.train_loader, learning_rate=1.0e-4, nb_epoch=1)
        self.training_losses.append(average_training_loss)
        return self.get_parameters(config={}),  self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, average_validation_loss = evaluate(DEVICE, self.model, self.val_loader)
        self.validation_losses.append(average_validation_loss)
        return float(loss),  self.num_examples["testset"], {}
    
    def graph(self):
        print(f"Training Losses: {self.training_losses}")
        print(f"Validation Losses: {self.validation_losses}")
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_losses, label='Training Loss')
        plt.plot(self.validation_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Client 1 (Stage 1) Training and Validation Losses')
        plt.legend()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    args = parser.parse_args()
    model = pytorchCNN(0.5).to(DEVICE)
    train_loader, val_loader, num_examples = load_data(args.data_dir, test_size=0.2, batch_size=40)
    client = FlowerClient(model, train_loader, val_loader, num_examples)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client.to_client())
    client.graph()

if __name__ == "__main__":
    main()