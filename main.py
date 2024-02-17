import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from utils import INPUT_SHAPE, batch_generator, preprocess_pytorch
import time

import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn import MSELoss

from model import cnn
from pytorch_model import pytorchCNN

#set seed
np.random.seed(0)

#constants
TRAIN_VAL_SPLIT = 0.75
LEARNING_RATE = 0.001
BATCH_SIZE = 32
SAMPLES_PER_EPOCH = 20000    
EPOCHS = 10
DATA_DIRECTORY = "C:\\Users\\User\\Self-Driving-Car\\data_new"

def load_data():
    data = pd.read_csv(os.path.join(DATA_DIRECTORY, 'driving_log.csv'))
    
    #Assign column names
    data.columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']

    #Change the file locations of image data
    #new_directory = "/mnt/c/Users/User/Self-Driving-Car/data/IMG"
    #data.iloc[:, :3] = data.iloc[:, :3].map(lambda x: rename_data(x, new_directory))

    X = data[['center', 'left', 'right']].values
    Y = data[['steering']].values
    #add in throttle as output for model
    NewY = data[['steering', 'throttle']].values
    V = data['speed'].values

    #Maybe create own shuffling method if scikit learn doesnt allow to return the reordered indexes

    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=TRAIN_VAL_SPLIT, random_state=0)
    print("Loaded data")
    return X_train, X_valid, Y_train, Y_valid

def rename_data(row, new_directory):
    return os.path.join(new_directory, row[40:])

def train_CNN(model, X_train, X_valid, Y_train, Y_valid):

    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min')
    
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=LEARNING_RATE))

    model.fit(batch_generator("data", X_train, Y_train, BATCH_SIZE, True),
                        steps_per_epoch=SAMPLES_PER_EPOCH,
                        epochs=EPOCHS,
                        max_queue_size=1,
                        validation_data=batch_generator("data", X_valid, Y_valid, BATCH_SIZE, False),
                        validation_steps=len(X_valid) // BATCH_SIZE,
                        callbacks=[checkpoint],
                        verbose=1)
    
def train_pytorchCNN(device, model, X_train, X_valid, Y_train, Y_valid):
    print("Training model")
    model = model.to(device)
    #Preprocess images for dataloaders
    X_train, Y_train = preprocess_pytorch("data_new", X_train, Y_train, len(X_train), True)
    X_valid, Y_valid = preprocess_pytorch("data_new", X_valid, Y_valid, len(X_valid), True)
    #Convert the training data to torch tensors
    X_train_tensor = torch.tensor(X_train).float().permute(0, 3, 1, 2)
    X_valid_tensor = torch.tensor(X_valid).float().permute(0, 3, 1, 2)
    Y_train_tensor = torch.tensor(Y_train).float()
    Y_valid_tensor = torch.tensor(Y_valid).float()

    #make it so that datasets combine the v variables
    #Create TensorDatasets for training and validation data
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, Y_valid_tensor)

    #Create DataLoaders for training and validation data
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Define the loss function and optimizer
    loss_function = MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    trainStart = time.time()
    #Training loop
    for epoch in range(EPOCHS):
        epochStart = time.time()
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            #Send input and target to device
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
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader)}')

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                #Send input and target to device
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                val_loss += loss.item()

        print(f'Validation Loss: {val_loss/len(valid_loader)}')
        epochTimeTaken = (epochEnd - epochStart) / 60
        print("time to train epoch: ", epochTimeTaken, " minutes\n")
    trainEnd = time.time()
    trainTimeTaken = (trainEnd - trainStart) / 60
    print("time to train: ", trainTimeTaken, " minutes\n")

    return model

def save_model(model, modelFileName):
    print("Saving model: ", modelFileName)
    folder = "models/"
    torch.save(model, folder+modelFileName)

def main():
    #print(tf.config.list_physical_devices('GPU'))
    
    X_train, X_valid, Y_train, Y_valid = load_data()
    #print(X_train[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    pytorch_cnn = pytorchCNN()
    pytorch_cnn = train_pytorchCNN(device, pytorch_cnn, X_train, X_valid, Y_train, Y_valid)
    save_model(pytorch_cnn, "augment(data_new)_pytorchCNN.h5")

    #model = cnn()

    #train_CNN(model, X_train, X_valid, Y_train, Y_valid)   

main()

