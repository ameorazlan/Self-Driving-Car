import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

INPUT_SHAPE = (3, 66, 200)

class pytorchCNN(nn.Module):
    def __init__(self):
        super(pytorchCNN, self).__init__()
        self.lambda_layer = LambdaLayer()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64*1*18, out_features=100)  # Adjust the in_features depending on the output of the last conv layer
        self.fc2 = nn.Linear(in_features=100, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=10)
        self.fc4 = nn.Linear(in_features=10, out_features=1)
        
    def forward(self, x):
        x = self.lambda_layer(x)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x

class LambdaLayer(nn.Module):
    def __init__(self):
        super(LambdaLayer, self).__init__()
    
    def forward(self, x):
        return x / 127.5 - 1.0

# Assuming INPUT_SHAPE is a tuple (channels, height, width)
model = pytorchCNN()
summary(model, INPUT_SHAPE)
