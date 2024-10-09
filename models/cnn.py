from torch import nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdjustableCNN(nn.Module):
    def __init__(self, depth, channel_num=24):
        super(AdjustableCNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.fc = nn.Linear(channel_num, 10)  # One FC layer with 32 input features and 10 output classes

        # Define the convolutional layers with adjustable depth
        for i in range(depth):
            if i == 0:
                self.conv_layers.append(nn.Conv2d(3, channel_num, 3, padding=1))
            else:
                self.conv_layers.append(nn.Conv2d(channel_num, channel_num, 3, padding=1))
            self.conv_layers.append(nn.BatchNorm2d(channel_num))
            self.conv_layers.append(nn.ReLU(inplace=True))
            self.conv_layers.append(nn.MaxPool2d(2, 2))

        # Adaptive pooling to calculate the final input size for the FC layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        #count = 0
        for layer in self.conv_layers:
            x = layer(x)
            # print(count)
            # print(x.shape)
            #count += 1
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

