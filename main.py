import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from preprocess import INPUT_SHAPE, batch_generator
import argparse
import os
import tensorflow as tf

from model import cnn

#set seed
np.random.seed(0)

#constants
TRAIN_VAL_SPLIT = 0.75
LEARNING_RATE = 0.001
BATCH_SIZE = 32
SAMPLES_PER_EPOCH = 20000    
EPOCHS = 10
DATA_DIRECTORY = "C:\\Users\\User\\Self-Driving-Car\\data"

def load_data():
    data = pd.read_csv(os.path.join(DATA_DIRECTORY, 'driving_log.csv'))
    
    #Assign column names
    data.columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']

    #Change the file locations of image data
    #new_directory = "/mnt/c/Users/User/Self-Driving-Car/data/IMG"
    #data.iloc[:, :3] = data.iloc[:, :3].map(lambda x: rename_data(x, new_directory))

    X = data[['center', 'left', 'right']].values
    Y = data[['steering']].values

    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=TRAIN_VAL_SPLIT, random_state=0)
    
    return X_train, X_valid, Y_train, Y_valid

def rename_data(row, new_directory):
    return os.path.join(new_directory, row[40:])

def train_CNN(model, X_train, X_valid, Y_train, Y_valid):

    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min')
    
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=LEARNING_RATE))

    model.fit(batch_generator("data", X_train, Y_train, BATCH_SIZE, True),
                        steps_per_epoch=SAMPLES_PER_EPOCH,
                        epochs=EPOCHS,
                        max_queue_size=1,
                        validation_data=batch_generator("data", X_valid, Y_valid, BATCH_SIZE, False),
                        validation_steps=len(X_valid) // BATCH_SIZE,
                        callbacks=[checkpoint],
                        verbose=1)

def main():
    #print(tf.config.list_physical_devices('GPU'))
    
    X_train, X_valid, Y_train, Y_valid = load_data()
    #print(X_train[0])

    model = cnn()

    train_CNN(model, X_train, X_valid, Y_train, Y_valid)
    

main()

