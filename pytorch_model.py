import torch.nn as nn
import torch.nn.functional as F
import torch

class pytorchCNN(nn.Module):
    def __init__(self, keep_prob):
        super(pytorchCNN, self).__init__()
        self.lambda_layer = LambdaLayer()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.dropout = nn.Dropout(keep_prob)
        self.flatten = nn.Flatten()
        # Dynamically calculate the size for the first fully connected layer
        conv_output_size = self._get_conv_output((3, 66, 200))
        print(conv_output_size)
        self.fc1 = nn.Linear(conv_output_size, 100)
        self.fc2 = nn.Linear(in_features=100, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=10)
        self.fc4 = nn.Linear(in_features=10, out_features=1)
        
    def forward(self, x):
            #x is tuple here
        #split x into x, v
        #x, v = x
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
        #pass in v only in last last layer
        x = self.fc4(x)
        return x
    
    def _get_conv_output(self, shape):
        batch_size = 1  # Use a dummy batch size of 1 for simplicity
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output = self._forward_features(input)
        n_size = output.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.conv1(x)
        x = F.elu(x)
        x = self.conv2(x)
        x = F.elu(x)
        x = self.conv3(x)
        x = F.elu(x)
        x = self.conv4(x)
        x = F.elu(x)
        x = self.conv5(x)
        x = F.elu(x)
        return x

class pytorchCNNSpeed(nn.Module):
    def __init__(self, keep_prob):
        super(pytorchCNNSpeed, self).__init__()
        self.lambda_layer = LambdaLayer()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.dropout = nn.Dropout(keep_prob)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1153, 101)
        self.fc2 = nn.Linear(in_features=101, out_features=51)
        self.fc3 = nn.Linear(in_features=51, out_features=11)
        self.fc4 = nn.Linear(in_features=11, out_features=1)
        
    def forward(self, x):
        #Split into image and speed
        x, v = x

        x = self.lambda_layer(x)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = self.dropout(x)
        x = self.flatten(x)

        x = torch.cat((x, v.unsqueeze(1)), dim=1)  
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        #pass in v only in last last layer
        x = self.fc4(x)
        return x


    def _forward_features(self, x):
        x = self.conv1(x)
        x = F.elu(x)
        x = self.conv2(x)
        x = F.elu(x)
        x = self.conv3(x)
        x = F.elu(x)
        x = self.conv4(x)
        x = F.elu(x)
        x = self.conv5(x)
        x = F.elu(x)
        return x
class LambdaLayer(nn.Module):
    def __init__(self):
        super(LambdaLayer, self).__init__()
    
    def forward(self, x):
        return x / 127.5 - 1.0

class pytorchCNNSpeedThrottle(nn.Module):
    def __init__(self, keep_prob):
        super(pytorchCNNSpeedThrottle, self).__init__()
        self.lambda_layer = LambdaLayer()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.dropout = nn.Dropout(keep_prob)
        self.flatten = nn.Flatten()


        self.fc1 = nn.Linear(1153, 101)  
        self.fc2 = nn.Linear(in_features=101, out_features=51)
        self.fc3 = nn.Linear(in_features=51, out_features=11)


        self.fc_steering = nn.Linear(in_features=11, out_features=1)
        self.fc_throttle = nn.Linear(in_features=11, out_features=1) 
        
    def forward(self, inputs):

        x, v = inputs

        x = self.lambda_layer(x)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = self.dropout(x)
        x = self.flatten(x)

        x = torch.cat((x, v.unsqueeze(1)), dim=1)
        
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))

        # Produce two outputs: steering and throttle
        steering = self.fc_steering(x)
        throttle = self.fc_throttle(x)
        
        return steering, throttle

class LambdaLayer(nn.Module):
    def __init__(self):
        super(LambdaLayer, self).__init__()
    
    def forward(self, x):
        return x / 127.5 - 1.0

