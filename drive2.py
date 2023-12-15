#decoding camera images
import base64
#for frametimestamp saving
from datetime import datetime
#reading and writing files
import os
#high level file operations
import shutil
#matrix math
import numpy as np
#real-time server
import socketio
#concurrent networking 
import eventlet
#web server gateway interface
import eventlet.wsgi
#image manipulation
from PIL import Image
#web framework
from flask import Flask
#input output
from io import BytesIO

#load our saved model
from keras.models import load_model

#helper class
import preprocess

import torch
import torchvision.transforms as transforms

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)
#init our model and image array as empty
model = None
prev_image_array = None

#set min/max speed for our autonomous car
MAX_SPEED = 25
MIN_SPEED = 10

#and a speed limit
speed_limit = MAX_SPEED

#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    print("In telemetry")
    if data:
        #Current steering angle of the car
        steering_angle = float(data["steering_angle"])
        #Current throttle of the car
        throttle = float(data["throttle"])
        #Current speed of the car
        speed = float(data["speed"])
        #Current center image of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        #Send control to the simulator
        try:
            image = np.asarray(image)       # from PIL image to numpy array
            image = preprocess.preprocess(image) # apply the preprocessing
            # Add batch dimension and convert to tensor
            image = torch.tensor(image).float().unsqueeze(0)
            # Permute to get the correct order (batch_size, channels, height, width)
            image = image.permute(0, 3, 1, 2)
            # Move the tensor to the same device as the model
            image = image.to(next(model.parameters()).device)

            # Predict the steering angle for the image
            with torch.no_grad():  # Ensure no gradients are calculated
                model.eval()  # Set the model to evaluation mode
                steering_angle = model(image).item()  # Get the prediction and convert to a Python float

            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)

        except Exception as e:
            print(e)

        # save frame
        if image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(image_folder, timestamp)
            # Assuming `image` is a PyTorch tensor with shape [1, channels, height, width]
            image = image.squeeze(0)  # Remove the batch dimension
            image = image.cpu().numpy()  # Convert to numpy array
            image = np.transpose(image, (1, 2, 0))  # Change from CHW to HWC format for PIL
            
            # Convert to uint8 (Check the tensor range to correctly scale it, e.g., if it's [-1, 1] or [0, 1])
            image = ((image * 0.5 + 0.5) * 255).astype(np.uint8)  # This scales and converts
            
            processed_image = Image.fromarray(image)  # Convert back to PIL image
            processed_image.save('{}.jpg'.format(image_filename))
    else:
        
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':

    #load model
    model = torch.load('C:\\Users\\User\\Self-Driving-Car\\models\\pytorchCNN.h5')
    #model = load_model(args.model)
    image_folder = 'C:\\Users\\User\\Self-Driving-Car\\test_data' 
    if image_folder != '':
        print("Creating image folder at {}".format(image_folder))
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        else:
            shutil.rmtree(image_folder)
            os.makedirs(image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)