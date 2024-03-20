import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator, batch_generator_with_speed_input, batch_generator_with_speed_throttle
import argparse
import os

np.random.seed(0) 

def load_data(args):
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
    return X_train, X_valid, y_train, y_valid

def build_model(args):
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    return model

def train_model(model, args, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint('latest_model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=args.learning_rate))
    model.fit(x=batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
              steps_per_epoch=args.samples_per_epoch // args.batch_size,
              epochs=args.nb_epoch,
              validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
              validation_steps=len(X_valid) // args.batch_size,
              callbacks=[checkpoint],
              verbose=1)

def s2b(s):
    s = s.lower()
    return s in ['true', 'yes', 'y', '1']

def load_data_with_speed_input(args):
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right', 'speed']].values
    y = data_df['steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
    return X_train, X_valid, y_train, y_valid

def build_model_with_speed_input(args):
    # Image input branch
    image_input = Input(shape=INPUT_SHAPE, name='image_input')
    x = Lambda(lambda x: x / 127.5 - 1.0)(image_input)
    x = Conv2D(24, (5, 5), activation='elu', strides=(2, 2))(x)
    x = Conv2D(36, (5, 5), activation='elu', strides=(2, 2))(x)
    x = Conv2D(48, (5, 5), activation='elu', strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='elu')(x)
    x = Conv2D(64, (3, 3), activation='elu')(x)
    x = Dropout(args.keep_prob)(x)
    x = Flatten()(x)
    x = Dense(100, activation='elu')(x)
    x = Dense(50, activation='elu')(x)
    x = Dense(10, activation='elu')(x)

    # Speed input branch
    speed_input = Input(shape=(1,), name='speed_input')
    
    # Combine the output of the last dense layer with the speed input
    combined = Concatenate()([x, speed_input])
    
    # Output layer
    steering_output = Dense(1, name='steering_output')(combined)
    
    # Create model
    model = Model(inputs=[image_input, speed_input], outputs=steering_output)
    model.summary()
    return model

def train_model_with_speed_input(model, args, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint('model_with_speed-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=args.learning_rate))
    model.fit(x=batch_generator_with_speed_input(args.data_dir, X_train, y_train, args.batch_size, True),
              steps_per_epoch=args.samples_per_epoch // args.batch_size,
              epochs=args.nb_epoch,
              validation_data=batch_generator_with_speed_input(args.data_dir, X_valid, y_valid, args.batch_size, False),
              validation_steps=len(X_valid) // args.batch_size,
              callbacks=[checkpoint],
              verbose=1)

def load_data_with_speed_throttle_input(args):
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), 
                          names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    
    # Inputs include images and speed
    X = data_df[['center', 'left', 'right', 'speed']].values
    
    # Outputs now include steering and throttle
    y = data_df[['steering', 'throttle']].values
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
    
    return X_train, X_valid, y_train, y_valid

def build_model_with_speed_throttle_input(args):
    # Image input branch
    image_input = Input(shape=INPUT_SHAPE, name='image_input')
    x = Lambda(lambda x: x / 127.5 - 1.0)(image_input)
    x = Conv2D(24, (5, 5), activation='elu', strides=(2, 2))(x)
    x = Conv2D(36, (5, 5), activation='elu', strides=(2, 2))(x)
    x = Conv2D(48, (5, 5), activation='elu', strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='elu')(x)
    x = Conv2D(64, (3, 3), activation='elu')(x)
    x = Dropout(args.keep_prob)(x)
    x = Flatten()(x)
    x = Dense(100, activation='elu')(x)
    x = Dense(50, activation='elu')(x)
    x = Dense(10, activation='elu')(x)

    # Speed input branch
    speed_input = Input(shape=(1,), name='speed_input')
    
    # Combine the output of the last dense layer with the speed input
    combined = Concatenate()([x, speed_input])
    
    # activation function of sigmoid and hyperbolic
    
    # Output layers for steering and throttle
    steering_output = Dense(1, name='steering_output')(combined)
    throttle_output = Dense(1, name='throttle_output')(combined)  # Additional output layer for throttle
    

    # Create model with multi-output
    model = Model(inputs=[image_input, speed_input], outputs=[steering_output, throttle_output])
    model.summary()
    return model

def train_model_with_speed_throttle_input(model, args, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint('model_with_speed_throttle-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=args.learning_rate))
    model.fit(x=batch_generator_with_speed_throttle(args.data_dir, X_train, y_train, args.batch_size, True),
              steps_per_epoch=args.samples_per_epoch // args.batch_size,
              epochs=args.nb_epoch,
              validation_data=batch_generator_with_speed_throttle(args.data_dir, X_valid, y_valid, args.batch_size, False),
              validation_steps=len(X_valid) // args.batch_size,
              callbacks=[checkpoint],
              verbose=1)

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

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #data = load_data(args)
    #model = build_model(args)
    #train_model(model, args, *data)

    #data_with_speed_input = load_data_with_speed_input(args)
    #model_with_speed_input = build_model_with_speed_input(args)
    #train_model_with_speed_input(model_with_speed_input, args, *data_with_speed_input)

    data_with_speed_throttle = load_data_with_speed_throttle_input(args)
    model_with_speed_throttle = build_model_with_speed_throttle_input(args)
    train_model_with_speed_throttle_input(model_with_speed_throttle, args, *data_with_speed_throttle)

if __name__ == '__main__':
    main()
