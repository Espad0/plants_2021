import torch.nn as nn
import torch.nn.functional as F


input_size = 32
number_of_maxpools = 2
fc_size = int(input_size / (number_of_maxpools * 2))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16*fc_size*fc_size, 12)
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x