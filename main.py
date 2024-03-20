import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from utils import preprocess_pytorch_tensor, preprocess_pytorch_speed, CustomDataset, preprocess_pytorch_speed_throttle
import time
import argparse
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn import MSELoss
from torchsummary import summary

from pytorch_model import pytorchCNN, pytorchCNNSpeed, pytorchCNNSpeedThrottle

#set seed
np.random.seed(0)

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)



def load_data(args):
    data = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))
    print(data.shape)
    #Assign column names
    data.columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']

    #Change the file locations of image data
    #new_directory = "/mnt/c/Users/User/Self-Driving-Car/data/IMG"
    #data.iloc[:, :3] = data.iloc[:, :3].map(lambda x: rename_data(x, new_directory))

    #X = data[['center', 'left', 'right', 'speed']].values
    X = data[['center', 'left', 'right']].values
    Y = data[['steering']].values
    #Y = data[['steering', 'throttle']].values

    #Maybe create own shuffling method if scikit learn doesnt allow to return the reordered indexes

    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=args.test_size, random_state=0)
    print("Loaded data")
    return X_train, X_valid, Y_train, Y_valid

def rename_data(row, new_directory):
    return os.path.join(new_directory, row[40:])
    
def train_pytorchCNN(device, model, X_train, X_valid, Y_train, Y_valid, args):

    print("Training model")

    model = model.to(device)
    #Preprocess images for dataloaders
    X_train, Y_train = preprocess_pytorch_tensor(X_train, Y_train, args.data_dir, True)
    X_valid, Y_valid = preprocess_pytorch_tensor(X_valid, Y_valid, args.data_dir, False)
    #(inputs_train, speeds_train), targets_train = preprocess_pytorch_speed(X_train, Y_train, args.data_dir, True)
    #(inputs_valid, speeds_valid), targets_valid= preprocess_pytorch_speed(X_valid, Y_valid, args.data_dir, False)
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    print("created data loaders")

    # Define the loss function and optimizer
    loss_function = MSELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    training_losses = []
    validation_losses = []

    trainStart = time.time()
    #Training loop
    print("starting training loop")
    for epoch in range(args.nb_epoch):
        counter = 0
        epochStart = time.time()
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            counter += 1
            #print(counter)

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
        epochEnd = time.time()
        print(f'Epoch {epoch+1}/{args.nb_epoch}, Loss: {running_loss/len(train_loader)}')

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
        epochTimeTaken = (epochEnd - epochStart) / 60
        print("time to train epoch: ", epochTimeTaken, " minutes\n")

        # Save model
        if (epoch + 1) % 5 == 0:
            print("Saving model")
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_model(state, "pytorch_model_speed")

        average_training_loss = running_loss / len(train_loader)
        average_validation_loss = val_loss / len(valid_loader)

        training_losses.append(average_training_loss)
        validation_losses.append(average_validation_loss)

    trainEnd = time.time()
    trainTimeTaken = (trainEnd - trainStart) / 60
    print("time to train: ", trainTimeTaken, " minutes\n")

    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Stage 1 Model Training and Validation Losses')
    plt.legend()
    plt.show()

def train_pytorchCNN_Speed_Throttle(device, model, X_train, X_valid, Y_train, Y_valid, args):

    print("Training model")

    model = model.to(device)
    #Preprocess images for dataloaders
    #X_train, Y_train = preprocess_pytorch_tensor(X_train, Y_train, args.data_dir, True)
    #X_valid, Y_valid = preprocess_pytorch_tensor(X_valid, Y_valid, args.data_dir, False)
    (inputs_train, speeds_train), targets_train = preprocess_pytorch_speed_throttle(X_train, Y_train, args.data_dir, True)
    (inputs_valid, speeds_valid), targets_valid= preprocess_pytorch_speed_throttle(X_valid, Y_valid, args.data_dir, False)
    print("Finished pre-processing")

    #Convert the training data to torch tensors
    #X_train_tensor = X_train #.permute(0, 3, 1, 2)
    #X_valid_tensor = X_valid #.permute(0, 3, 1, 2)
    #Y_train_tensor = Y_train
    #Y_valid_tensor = Y_valid
    print("converted into tensors")

    #make it so that datasets combine the v variables
    #Create TensorDatasets for training and validation data
    #train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    #valid_dataset = TensorDataset(X_valid_tensor, Y_valid_tensor)
    train_dataset = CustomDataset((inputs_train, speeds_train), targets_train)
    valid_dataset = CustomDataset((inputs_valid, speeds_valid), targets_valid)
    print("created dataset")

    #Create DataLoaders for training and validation data
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    print("created data loaders")

    # Define the loss function and optimizer
    loss_function = MSELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    training_steering_losses = []
    training_throttle_losses = []
    validation_steering_losses = []
    validation_throttle_losses = []
    trainStart = time.time()
    #Training loop
    print("starting training loop")
    for epoch in range(args.nb_epoch):
        counter = 0
        epochStart = time.time()
        model.train()
        running_steering_loss = 0.0
        running_throttle_loss = 0.0
        for inputs, targets in train_loader:
            counter += 1
            #print(counter)
            #Send input and target to device
            inputs = tuple(input_tensor.to(device) for input_tensor in inputs)
            targets = targets.to(device)
            #inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            steering_outputs, throttle_outputs = model(inputs)

            steering_targets, throttle_targets = targets[:,0], targets[:,1]

            loss_steering = loss_function(steering_outputs, steering_targets.unsqueeze(1))
            loss_throttle = loss_function(throttle_outputs, throttle_targets.unsqueeze(1))
            #loss = loss_function(outputs, targets)
        
            loss = loss_steering + loss_throttle
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_steering_loss += loss_steering.item()
            running_throttle_loss += loss_throttle.item()
        epochEnd = time.time()
        print(f'Epoch {epoch+1}/{args.nb_epoch}, Steering Loss: {running_steering_loss/len(train_loader)}, Throttle Loss: {running_throttle_loss/len(train_loader)}')

        # Validation loop
        model.eval()
        val_loss_steering = 0.0
        val_loss_throttle = 0.0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs = tuple(input_tensor.to(device) for input_tensor in inputs)
                targets = targets.to(device)
                
                steering_output, throttle_output = model(inputs)
                targets_steering, targets_throttle = targets[:, 0], targets[:, 1]
                
                loss_steering = loss_function(steering_output, targets_steering.unsqueeze(1))
                loss_throttle = loss_function(throttle_output, targets_throttle.unsqueeze(1))
                
                val_loss_steering += loss_steering.item()
                val_loss_throttle += loss_throttle.item()

        print(f'Validation Steering Loss: {val_loss_steering/len(valid_loader)}, Validation Throttle Loss: {val_loss_throttle/len(valid_loader)},')
        epochTimeTaken = (epochEnd - epochStart) / 60
        print("time to train epoch: ", epochTimeTaken, " minutes\n")

                # At the end of each training epoch
        average_training_steering_loss = running_steering_loss / len(train_loader)
        average_training_throttle_loss = running_throttle_loss / len(train_loader)
        training_steering_losses.append(average_training_steering_loss)
        training_throttle_losses.append(average_training_throttle_loss)

        # At the end of each validation epoch
        average_validation_steering_loss = val_loss_steering / len(valid_loader)
        average_validation_throttle_loss = val_loss_throttle / len(valid_loader)
        validation_steering_losses.append(average_validation_steering_loss)
        validation_throttle_losses.append(average_validation_throttle_loss)

        # Save model
        """if (epoch + 1) % 5 == 0:
            print("Saving model")
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_model(state, "pytorch_model_speed_throttle)
        """
    trainEnd = time.time()
    trainTimeTaken = (trainEnd - trainStart) / 60
    print("time to train: ", trainTimeTaken, " minutes\n")
        # Plotting the losses
    plt.figure(figsize=(12, 6))

    plt.figure(figsize=(10, 6))

    # Plot training losses
    plt.plot(training_steering_losses, label='Training Steering Loss', color='blue', linestyle='dashed')
    plt.plot(training_throttle_losses, label='Training Throttle Loss', color='red', linestyle='dashed')

    # Plot validation losses
    plt.plot(validation_steering_losses, label='Validation Steering Loss', color='blue', marker='o', linestyle='None')
    plt.plot(validation_throttle_losses, label='Validation Throttle Loss', color='red', marker='x', linestyle='None')

    plt.title('Stage 3  Model Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def save_model(state, name):
    torch.save(state, "{}-{}.h5".format(name, state['epoch']))

def s2b(s):
    s = s.lower()
    return s in ['true', 'yes', 'y', '1']

def main():
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=15329)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()
    
    X_train, X_valid, Y_train, Y_valid = load_data(args)
    #print(X_train[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    pytorch_cnn = pytorchCNN(args.keep_prob)
    #pytorch_cnn = pytorchCNNSpeed(args.keep_prob)
    #pytorch_cnn = pytorchCNNSpeedThrottle(args.keep_prob)
    pytorch_cnn.to(device)
    #summary(pytorch_cnn, INPUT_SHAPE)
    train_pytorchCNN(device, pytorch_cnn, X_train, X_valid, Y_train, Y_valid, args)
    #train_pytorchCNN_Speed_Throttle(device, pytorch_cnn, X_train, X_valid, Y_train, Y_valid, args)

if __name__ == '__main__':
    main()
