import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from preprocess import INPUT_SHAPE, batch_generator
import argparse
import os

from model import cnn

#set seed
np.random.seed(0)

#constants
TRAIN_VAL_SPLIT = 0.75
LEARNING_RATE = 0.001
BATCH_SIZE = 32
SAMPLES_PER_EPOCH = 2000    
EPOCHS = 10

def load_data():
    data = pd.read_csv(os.path.join(os.getcwd(), 'driving_data.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse'])
    X = data[['center', 'left', 'right']].values
    Y = data[['steering']].values

    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=TRAIN_VAL_SPLIT, random_state=0)
    
    return X_train, X_valid, Y_train, Y_valid

def train_CNN(model, X_train, X_valid, Y_train, Y_valid):

    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_acc',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='max')
    
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=LEARNING_RATE))

    model.fit_generator(batch_generator("data", X_train, Y_train, BATCH_SIZE, True),
                        SAMPLES_PER_EPOCH,
                        EPOCHS,
                        max_q_size=1,
                        validation_data=batch_generator("data", X_valid, Y_valid, BATCH_SIZE, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)

def main():
    X_train, X_valid, Y_train, Y_valid = load_data()

    model = cnn()

    train_CNN(model, X_train, X_valid, Y_train, Y_valid)

main()

