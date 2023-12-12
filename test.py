import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '1' to show more warnings
import tensorflow as tf
from tensorflow.python.client import device_lib

local_devices = device_lib.list_local_devices()
gpus = [device.name for device in local_devices if device.device_type == 'GPU']
print("Available GPUs:", gpus)
